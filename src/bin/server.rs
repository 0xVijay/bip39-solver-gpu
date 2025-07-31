use std::env;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::Arc;

// Import from the local crate
use bip39_solver_gpu::config::Config;
use bip39_solver_gpu::job_server::JobServer;
use bip39_solver_gpu::job_types::*;

/// Simple HTTP server for the job server
/// In production, you would use a proper web framework like warp, axum, or actix-web
struct SimpleHttpServer {
    job_server: Arc<JobServer>,
    secret: String,
}

impl SimpleHttpServer {
    fn new(job_server: Arc<JobServer>, secret: String) -> Self {
        Self { job_server, secret }
    }

    /// Start the HTTP server
    fn run(&self, port: u16) -> Result<(), Box<dyn std::error::Error>> {
        let listener = TcpListener::bind(format!("0.0.0.0:{}", port))?;
        println!("Job server listening on port {}", port);

        for stream in listener.incoming() {
            match stream {
                Ok(stream) => {
                    let job_server = Arc::clone(&self.job_server);
                    let secret = self.secret.clone();

                    std::thread::spawn(move || {
                        if let Err(e) = handle_connection(stream, job_server, secret) {
                            eprintln!("Error handling connection: {}", e);
                        }
                    });
                }
                Err(e) => {
                    eprintln!("Error accepting connection: {}", e);
                }
            }
        }

        Ok(())
    }
}

/// Handle a single HTTP connection
fn handle_connection(
    mut stream: TcpStream,
    job_server: Arc<JobServer>,
    secret: String,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buf_reader = BufReader::new(&mut stream);

    // Read request line
    let mut request_line = String::new();
    buf_reader.read_line(&mut request_line)?;
    let request_line = request_line.trim();

    // Parse HTTP request line
    let parts: Vec<&str> = request_line.split_whitespace().collect();
    if parts.len() < 3 {
        send_response(&mut stream, 400, "Bad Request", b"Invalid request line")?;
        return Ok(());
    }

    let method = parts[0];
    let path = parts[1];

    // Read headers to get content length and authorization
    let mut headers = std::collections::HashMap::new();
    loop {
        let mut line = String::new();
        buf_reader.read_line(&mut line)?;
        let line = line.trim();
        if line.is_empty() {
            break;
        }
        if let Some(colon_pos) = line.find(':') {
            let key = line[..colon_pos].trim().to_lowercase();
            let value = line[colon_pos + 1..].trim().to_string();
            headers.insert(key, value);
        }
    }

    // Check authorization
    if let Some(auth_header) = headers.get("authorization") {
        let expected = format!("Bearer {}", secret);
        if auth_header != &expected {
            send_response(&mut stream, 401, "Unauthorized", b"Invalid authorization")?;
            return Ok(());
        }
    } else {
        send_response(&mut stream, 401, "Unauthorized", b"Authorization required")?;
        return Ok(());
    }

    // Route requests
    match (method, path) {
        ("POST", "/api/jobs/request") => {
            handle_job_request(&mut stream, job_server, &headers)?;
        }
        ("POST", "/api/jobs/complete") => {
            handle_job_completion(&mut stream, job_server, &headers)?;
        }
        ("POST", "/api/jobs/heartbeat") => {
            handle_heartbeat(&mut stream, job_server, &headers)?;
        }
        ("GET", "/api/status") => {
            handle_status_request(&mut stream, job_server)?;
        }
        _ => {
            send_response(&mut stream, 404, "Not Found", b"Endpoint not found")?;
        }
    }

    Ok(())
}

/// Handle job request from worker
fn handle_job_request(
    stream: &mut TcpStream,
    job_server: Arc<JobServer>,
    headers: &std::collections::HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Read request body
    let content_length: usize = headers
        .get("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if content_length == 0 {
        send_response(stream, 400, "Bad Request", b"Missing request body")?;
        return Ok(());
    }

    let mut body = vec![0; content_length];
    stream.read_exact(&mut body)?;

    let request: JobRequest = serde_json::from_slice(&body)?;

    match job_server.assign_job(&request) {
        Ok(response) => {
            let json = serde_json::to_string(&response)?;
            send_json_response(stream, 200, "OK", json.as_bytes())?;
        }
        Err(api_error) => {
            let json = serde_json::to_string(&api_error)?;
            send_json_response(stream, api_error.code, "Error", json.as_bytes())?;
        }
    }

    Ok(())
}

/// Handle job completion from worker
fn handle_job_completion(
    stream: &mut TcpStream,
    job_server: Arc<JobServer>,
    headers: &std::collections::HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let content_length: usize = headers
        .get("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if content_length == 0 {
        send_response(stream, 400, "Bad Request", b"Missing request body")?;
        return Ok(());
    }

    let mut body = vec![0; content_length];
    stream.read_exact(&mut body)?;

    let completion: JobCompletion = serde_json::from_slice(&body)?;

    match job_server.complete_job(&completion) {
        Ok(()) => {
            send_response(stream, 200, "OK", b"Job completion recorded")?;
        }
        Err(api_error) => {
            let json = serde_json::to_string(&api_error)?;
            send_json_response(stream, api_error.code, "Error", json.as_bytes())?;
        }
    }

    Ok(())
}

/// Handle heartbeat from worker
fn handle_heartbeat(
    stream: &mut TcpStream,
    job_server: Arc<JobServer>,
    headers: &std::collections::HashMap<String, String>,
) -> Result<(), Box<dyn std::error::Error>> {
    let content_length: usize = headers
        .get("content-length")
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if content_length == 0 {
        send_response(stream, 400, "Bad Request", b"Missing request body")?;
        return Ok(());
    }

    let mut body = vec![0; content_length];
    stream.read_exact(&mut body)?;

    let heartbeat: WorkerHeartbeat = serde_json::from_slice(&body)?;

    match job_server.update_heartbeat(&heartbeat) {
        Ok(()) => {
            send_response(stream, 200, "OK", b"Heartbeat updated")?;
        }
        Err(api_error) => {
            let json = serde_json::to_string(&api_error)?;
            send_json_response(stream, api_error.code, "Error", json.as_bytes())?;
        }
    }

    Ok(())
}

/// Handle status request
fn handle_status_request(
    stream: &mut TcpStream,
    job_server: Arc<JobServer>,
) -> Result<(), Box<dyn std::error::Error>> {
    let status = job_server.get_status();
    let json = serde_json::to_string(&status)?;
    send_json_response(stream, 200, "OK", json.as_bytes())?;

    Ok(())
}

/// Send HTTP response
fn send_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    body: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Length: {}\r\n\r\n",
        status_code,
        status_text,
        body.len()
    );

    stream.write_all(response.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

/// Send JSON response
fn send_json_response(
    stream: &mut TcpStream,
    status_code: u16,
    status_text: &str,
    body: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let response = format!(
        "HTTP/1.1 {} {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n",
        status_code,
        status_text,
        body.len()
    );

    stream.write_all(response.as_bytes())?;
    stream.write_all(body)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() != 3 || args[1] != "--config" {
        eprintln!("Usage: {} --config <config.json>", args[0]);
        std::process::exit(1);
    }

    let config_path = &args[2];
    let config = Config::load(config_path)?;

    println!("Starting job server with config from: {}", config_path);
    println!("Target address: {}", config.ethereum.target_address);

    // Get secret for authentication
    let secret = config
        .worker
        .as_ref()
        .map(|w| w.secret.clone())
        .unwrap_or_else(|| "default-secret".to_string());

    // Create and initialize job server
    let job_server = Arc::new(JobServer::new(config)?);
    job_server.initialize_jobs()?;

    // Start timeout handler
    job_server.start_timeout_handler();

    // Start HTTP server
    let http_server = SimpleHttpServer::new(job_server, secret);
    http_server.run(3000)?;

    Ok(())
}
