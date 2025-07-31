use crate::config::SlackConfig;
use reqwest;
use serde_json::json;

pub struct SlackNotifier {
    config: SlackConfig,
    client: reqwest::blocking::Client,
}

impl SlackNotifier {
    pub fn new(config: SlackConfig) -> Self {
        Self {
            config,
            client: reqwest::blocking::Client::new(),
        }
    }

    /// Send a notification when a mnemonic is found
    pub fn notify_solution_found(
        &self,
        mnemonic: &str,
        address: &str,
        offset: u128,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let message = format!(
            "🎉 *Mnemonic Found!* 🎉\n\
            \n\
            **Mnemonic:** `{}`\n\
            **Address:** `{}`\n\
            **Offset:** `{}`\n\
            \n\
            The search has been completed successfully!",
            mnemonic, address, offset
        );

        self.send_message(&message, Some("good"))
    }

    /// Send a notification about work progress
    pub fn notify_progress(
        &self,
        offset: u128,
        rate: f64,
        elapsed_time: u64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let message = format!(
            "⚡ *Search Progress Update* ⚡\n\
            \n\
            **Current Offset:** `{}`\n\
            **Search Rate:** `{:.2} mnemonics/sec`\n\
            **Elapsed Time:** `{} seconds`\n\
            \n\
            Search is ongoing...",
            offset, rate, elapsed_time
        );

        self.send_message(&message, Some("warning"))
    }

    /// Send a notification when search starts
    pub fn notify_search_started(
        &self,
        target_address: &str,
        total_combinations: u128,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let message = format!(
            "🚀 *Ethereum Mnemonic Search Started* 🚀\n\
            \n\
            **Target Address:** `{}`\n\
            **Total Combinations:** `{}`\n\
            \n\
            Beginning GPU-accelerated search...",
            target_address, total_combinations
        );

        self.send_message(&message, Some("good"))
    }

    /// Send a notification when an error occurs
    pub fn notify_error(&self, error_message: &str) -> Result<(), Box<dyn std::error::Error>> {
        let message = format!(
            "❌ *Error Occurred* ❌\n\
            \n\
            **Error:** `{}`\n\
            \n\
            Please check the logs for more details.",
            error_message
        );

        self.send_message(&message, Some("danger"))
    }

    /// Send a custom message to Slack
    fn send_message(
        &self,
        text: &str,
        color: Option<&str>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut payload = json!({
            "text": text,
        });

        // Add channel if specified
        if let Some(channel) = &self.config.channel {
            payload["channel"] = json!(channel);
        }

        // Add color formatting if specified
        if let Some(color) = color {
            payload["attachments"] = json!([{
                "color": color,
                "text": text,
            }]);
            // Remove the top-level text since it's in the attachment
            payload["text"] = json!("");
        }

        let response = self
            .client
            .post(&self.config.webhook_url)
            .json(&payload)
            .send()?;

        if !response.status().is_success() {
            return Err(format!(
                "Slack notification failed with status: {}",
                response.status()
            )
            .into());
        }

        Ok(())
    }
}

/// Send a simple Slack notification without creating a SlackNotifier instance
pub fn send_simple_notification(
    webhook_url: &str,
    message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let client = reqwest::blocking::Client::new();
    let payload = json!({
        "text": message,
    });

    let response = client.post(webhook_url).json(&payload).send()?;

    if !response.status().is_success() {
        return Err(format!(
            "Slack notification failed with status: {}",
            response.status()
        )
        .into());
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_slack_config_creation() {
        let config = SlackConfig {
            webhook_url: "https://hooks.slack.com/services/test".to_string(),
            channel: Some("#test".to_string()),
        };

        let notifier = SlackNotifier::new(config);
        // Just test that creation works - we can't test actual notifications without a real webhook
        assert!(!notifier.config.webhook_url.is_empty());
    }

    #[test]
    fn test_message_formatting() {
        // Test that our message formatting doesn't panic
        let config = SlackConfig {
            webhook_url: "https://hooks.slack.com/services/test".to_string(),
            channel: None,
        };

        let notifier = SlackNotifier::new(config);

        // These will fail to send but should not panic on message formatting
        let _ = notifier.notify_solution_found(
            "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about",
            "0x742d35Cc6634C0532925a3b8D581C027BD5b7c4f",
            12345
        );

        let _ = notifier.notify_progress(12345, 1000.5, 3600);
        let _ = notifier.notify_error("Test error message");
    }
}
