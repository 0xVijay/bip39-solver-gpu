/// Utility functions for hex formatting, checksums, and logging
use std::time::{SystemTime, UNIX_EPOCH};

/// Hex encoding and decoding utilities
pub struct HexUtils;

impl HexUtils {
    /// Convert bytes to hex string with 0x prefix
    pub fn bytes_to_hex(bytes: &[u8]) -> String {
        format!("0x{}", hex::encode(bytes))
    }
    
    /// Convert bytes to hex string without 0x prefix
    pub fn bytes_to_hex_no_prefix(bytes: &[u8]) -> String {
        hex::encode(bytes)
    }
    
    /// Convert hex string to bytes (with or without 0x prefix)
    pub fn hex_to_bytes(hex_str: &str) -> Result<Vec<u8>, hex::FromHexError> {
        let clean_hex = if hex_str.starts_with("0x") {
            &hex_str[2..]
        } else {
            hex_str
        };
        hex::decode(clean_hex)
    }
    
    /// Validate hex string format
    pub fn is_valid_hex(hex_str: &str) -> bool {
        let clean_hex = if hex_str.starts_with("0x") {
            &hex_str[2..]
        } else {
            hex_str
        };
        
        !clean_hex.is_empty() && clean_hex.chars().all(|c| c.is_ascii_hexdigit())
    }
}

/// Checksum validation utilities
pub struct ChecksumUtils;

impl ChecksumUtils {
    /// Calculate simple checksum (XOR of all bytes)
    pub fn simple_checksum(data: &[u8]) -> u8 {
        data.iter().fold(0u8, |acc, &byte| acc ^ byte)
    }
    
    /// Validate data with appended checksum
    pub fn validate_with_checksum(data_with_checksum: &[u8]) -> bool {
        if data_with_checksum.is_empty() {
            return false;
        }
        
        let data_len = data_with_checksum.len() - 1;
        let data = &data_with_checksum[..data_len];
        let expected_checksum = data_with_checksum[data_len];
        
        Self::simple_checksum(data) == expected_checksum
    }
    
    /// Append checksum to data
    pub fn append_checksum(data: &[u8]) -> Vec<u8> {
        let mut result = data.to_vec();
        let checksum = Self::simple_checksum(data);
        result.push(checksum);
        result
    }
}

/// Logging utilities
pub struct Logger;

impl Logger {
    /// Log info message with timestamp
    pub fn info(message: &str) {
        println!("[INFO] {} - {}", Self::timestamp(), message);
    }
    
    /// Log warning message with timestamp
    pub fn warn(message: &str) {
        println!("[WARN] {} - {}", Self::timestamp(), message);
    }
    
    /// Log error message with timestamp
    pub fn error(message: &str) {
        eprintln!("[ERROR] {} - {}", Self::timestamp(), message);
    }
    
    /// Log debug message with timestamp (only in debug builds)
    pub fn debug(message: &str) {
        #[cfg(debug_assertions)]
        println!("[DEBUG] {} - {}", Self::timestamp(), message);
    }
    
    /// Get current timestamp as string
    pub fn timestamp() -> String {
        match SystemTime::now().duration_since(UNIX_EPOCH) {
            Ok(duration) => {
                let secs = duration.as_secs();
                let nanos = duration.subsec_nanos();
                format!("{}.{:03}", secs, nanos / 1_000_000)
            }
            Err(_) => "0.000".to_string(),
        }
    }
    
    /// Format duration in human-readable form
    pub fn format_duration(duration: std::time::Duration) -> String {
        let total_secs = duration.as_secs();
        let millis = duration.subsec_millis();
        
        if total_secs < 60 {
            format!("{}.{:03}s", total_secs, millis)
        } else if total_secs < 3600 {
            let mins = total_secs / 60;
            let secs = total_secs % 60;
            format!("{}m {}.{:03}s", mins, secs, millis)
        } else if total_secs < 86400 {
            let hours = total_secs / 3600;
            let mins = (total_secs % 3600) / 60;
            let secs = total_secs % 60;
            format!("{}h {}m {}s", hours, mins, secs)
        } else {
            let days = total_secs / 86400;
            let hours = (total_secs % 86400) / 3600;
            format!("{}d {}h", days, hours)
        }
    }
    
    /// Format large numbers with thousands separators
    pub fn format_number(number: u128) -> String {
        let num_str = number.to_string();
        let mut result = String::new();
        let chars: Vec<char> = num_str.chars().collect();
        
        for (i, &ch) in chars.iter().enumerate() {
            if i > 0 && (chars.len() - i) % 3 == 0 {
                result.push(',');
            }
            result.push(ch);
        }
        
        result
    }
    
    /// Format rate as human-readable
    pub fn format_rate(rate: f64, unit: &str) -> String {
        if rate >= 1_000_000.0 {
            format!("{:.2}M {}/s", rate / 1_000_000.0, unit)
        } else if rate >= 1_000.0 {
            format!("{:.2}K {}/s", rate / 1_000.0, unit)
        } else {
            format!("{:.2} {}/s", rate, unit)
        }
    }
}

/// Performance monitoring utilities
pub struct PerfUtils;

impl PerfUtils {
    /// Time a function execution
    pub fn time_function<T, F>(func: F) -> (T, std::time::Duration)
    where
        F: FnOnce() -> T,
    {
        let start = std::time::Instant::now();
        let result = func();
        let duration = start.elapsed();
        (result, duration)
    }
    
    /// Calculate progress percentage
    pub fn progress_percentage(current: u128, total: u128) -> f64 {
        if total == 0 {
            0.0
        } else {
            (current as f64 / total as f64) * 100.0
        }
    }
    
    /// Estimate time remaining
    pub fn estimate_eta(current: u128, total: u128, elapsed: std::time::Duration) -> std::time::Duration {
        if current == 0 {
            return std::time::Duration::from_secs(0);
        }
        
        let rate = current as f64 / elapsed.as_secs_f64();
        let remaining = total - current;
        let eta_seconds = if rate > 0.0 {
            remaining as f64 / rate
        } else {
            0.0
        };
        
        std::time::Duration::from_secs_f64(eta_seconds)
    }
    
    /// Create progress bar string
    pub fn progress_bar(current: u128, total: u128, width: usize) -> String {
        if total == 0 {
            return "█".repeat(width);
        }
        
        let progress = (current as f64 / total as f64).min(1.0);
        let filled = (progress * width as f64) as usize;
        let empty = width - filled;
        
        format!("{}{}",
            "█".repeat(filled),
            "░".repeat(empty)
        )
    }
}

/// Configuration validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Validate Ethereum address format
    pub fn is_valid_ethereum_address(address: &str) -> bool {
        if !address.starts_with("0x") || address.len() != 42 {
            return false;
        }
        
        let hex_part = &address[2..];
        hex_part.chars().all(|c| c.is_ascii_hexdigit())
    }
    
    /// Validate BIP44 derivation path format
    pub fn is_valid_derivation_path(path: &str) -> bool {
        if !path.starts_with("m/") {
            return false;
        }
        
        let parts: Vec<&str> = path.split('/').collect();
        if parts.len() < 2 {
            return false;
        }
        
        // Check each part after 'm'
        for part in &parts[1..] {
            if part.ends_with('\'') {
                // Hardened derivation
                let index_str = &part[..part.len() - 1];
                if index_str.parse::<u32>().is_err() {
                    return false;
                }
            } else {
                // Non-hardened derivation
                if part.parse::<u32>().is_err() {
                    return false;
                }
            }
        }
        
        true
    }
    
    /// Validate word list for BIP39
    pub fn validate_word_constraints(words: &[String]) -> Result<(), String> {
        use bip39::Language;
        let wordlist = Language::English.word_list();
        
        for word in words {
            if !wordlist.contains(&word.as_str()) {
                return Err(format!("Invalid BIP39 word: {}", word));
            }
        }
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_hex_utils() {
        let bytes = [0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef];
        let hex = HexUtils::bytes_to_hex(&bytes);
        assert_eq!(hex, "0x0123456789abcdef");
        
        let decoded = HexUtils::hex_to_bytes(&hex).unwrap();
        assert_eq!(decoded, bytes);
        
        assert!(HexUtils::is_valid_hex("0x123abc"));
        assert!(HexUtils::is_valid_hex("123ABC"));
        assert!(!HexUtils::is_valid_hex("0x123g"));
    }
    
    #[test]
    fn test_checksum_utils() {
        let data = [1, 2, 3, 4, 5];
        let checksum = ChecksumUtils::simple_checksum(&data);
        assert_eq!(checksum, 1 ^ 2 ^ 3 ^ 4 ^ 5);
        
        let data_with_checksum = ChecksumUtils::append_checksum(&data);
        assert!(ChecksumUtils::validate_with_checksum(&data_with_checksum));
    }
    
    #[test]
    fn test_logger() {
        Logger::info("Test info message");
        Logger::warn("Test warning message");
        Logger::debug("Test debug message");
        
        let duration = std::time::Duration::from_millis(1234);
        let formatted = Logger::format_duration(duration);
        assert_eq!(formatted, "1.234s");
        
        assert_eq!(Logger::format_number(1234567), "1,234,567");
        
        let rate = Logger::format_rate(1500.0, "ops");
        assert_eq!(rate, "1.50K ops/s");
    }
    
    #[test]
    fn test_perf_utils() {
        let (result, duration) = PerfUtils::time_function(|| {
            std::thread::sleep(std::time::Duration::from_millis(10));
            42
        });
        
        assert_eq!(result, 42);
        assert!(duration.as_millis() >= 10);
        
        let progress = PerfUtils::progress_percentage(25, 100);
        assert_eq!(progress, 25.0);
        
        let bar = PerfUtils::progress_bar(50, 100, 20);
        assert_eq!(bar.len(), 20);
    }
    
    #[test]
    fn test_validation_utils() {
        assert!(ValidationUtils::is_valid_ethereum_address("0x1234567890123456789012345678901234567890"));
        assert!(!ValidationUtils::is_valid_ethereum_address("1234567890123456789012345678901234567890"));
        assert!(!ValidationUtils::is_valid_ethereum_address("0x123"));
        
        assert!(ValidationUtils::is_valid_derivation_path("m/44'/60'/0'/0/0"));
        assert!(ValidationUtils::is_valid_derivation_path("m/0"));
        assert!(!ValidationUtils::is_valid_derivation_path("44'/60'/0'/0/0"));
        assert!(!ValidationUtils::is_valid_derivation_path("m/44x/60'/0'/0/0"));
    }
}