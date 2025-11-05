///! File-based monitoring for vLLM Metrics Collector integration
///!
///! This module monitors the pending queue directory created by the Python
///! Metrics Collector and submits metrics to the BFT Coordinator.
///!
///! Directory structure:
///!   base_dir/
///!     pending/
///!       req-123.jsonl  <- Read these
///!       req-456.jsonl
///!     sent/
///!       req-123_20251105_123456.jsonl  <- Move here after success
///!     failed/
///!       req-789_20251105_123500.jsonl  <- Move here after failure

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use tokio::time::{sleep, Duration};
use tracing::{debug, error, info, warn};

use crate::client::CoordinatorClient;
use crate::tpm::SimulatedTpmAgent;
use bft_common::RequestMetrics;

/// File monitor configuration
#[derive(Debug, Clone)]
pub struct FileMonitorConfig {
    /// Base directory containing pending/sent/failed subdirectories
    pub base_dir: PathBuf,
    /// Polling interval (seconds)
    pub poll_interval_secs: u64,
    /// Maximum files to process per iteration
    pub batch_size: usize,
    /// Async mode for submissions
    pub async_mode: bool,
}

impl Default for FileMonitorConfig {
    fn default() -> Self {
        Self {
            base_dir: PathBuf::from("/var/log/vllm/metrics"),
            poll_interval_secs: 1,
            batch_size: 10,
            async_mode: true,  // Default to async for better performance
        }
    }
}

/// File monitor that processes pending metrics
pub struct FileMonitor {
    config: FileMonitorConfig,
    pending_dir: PathBuf,
    sent_dir: PathBuf,
    failed_dir: PathBuf,
}

impl FileMonitor {
    /// Create a new file monitor
    pub fn new(config: FileMonitorConfig) -> Result<Self> {
        let pending_dir = config.base_dir.join("pending");
        let sent_dir = config.base_dir.join("sent");
        let failed_dir = config.base_dir.join("failed");

        // Ensure directories exist
        fs::create_dir_all(&pending_dir)
            .context("Failed to create pending directory")?;
        fs::create_dir_all(&sent_dir)
            .context("Failed to create sent directory")?;
        fs::create_dir_all(&failed_dir)
            .context("Failed to create failed directory")?;

        info!("File monitor initialized:");
        info!("  Pending: {}", pending_dir.display());
        info!("  Sent: {}", sent_dir.display());
        info!("  Failed: {}", failed_dir.display());

        Ok(Self {
            config,
            pending_dir,
            sent_dir,
            failed_dir,
        })
    }

    /// Run the file monitor loop
    pub async fn run(
        &self,
        client: &mut CoordinatorClient,
        tpm_agent: &SimulatedTpmAgent,
    ) -> Result<()> {
        info!("Starting file monitor loop (poll interval: {}s)", self.config.poll_interval_secs);
        info!("Verification mode: {}", if self.config.async_mode { "async" } else { "sync" });

        loop {
            match self.process_pending_files(client, tpm_agent).await {
                Ok(processed) => {
                    if processed > 0 {
                        info!("Processed {} pending files", processed);
                    } else {
                        debug!("No pending files found");
                    }
                }
                Err(e) => {
                    error!("Error processing pending files: {}", e);
                }
            }

            sleep(Duration::from_secs(self.config.poll_interval_secs)).await;
        }
    }

    /// Process all pending files
    async fn process_pending_files(
        &self,
        client: &mut CoordinatorClient,
        tpm_agent: &SimulatedTpmAgent,
    ) -> Result<usize> {
        let pending_files = self.get_pending_files()?;

        if pending_files.is_empty() {
            return Ok(0);
        }

        let to_process = std::cmp::min(pending_files.len(), self.config.batch_size);
        debug!("Found {} pending files, processing {}", pending_files.len(), to_process);

        let mut processed = 0;

        for file_path in pending_files.iter().take(to_process) {
            match self.process_single_file(file_path, client, tpm_agent).await {
                Ok(()) => {
                    processed += 1;
                }
                Err(e) => {
                    error!("Failed to process file {:?}: {}", file_path, e);
                    // Continue processing other files
                }
            }
        }

        Ok(processed)
    }

    /// Process a single pending file
    async fn process_single_file(
        &self,
        file_path: &Path,
        client: &mut CoordinatorClient,
        tpm_agent: &SimulatedTpmAgent,
    ) -> Result<()> {
        // Extract request_id from filename
        let file_name = file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .ok_or_else(|| anyhow::anyhow!("Invalid filename"))?;

        let request_id = file_name.to_string();

        debug!("Processing file: {} (request_id: {})", file_path.display(), request_id);

        // Read and parse metrics
        let metrics = self.read_metrics_file(file_path)
            .with_context(|| format!("Failed to read metrics from {:?}", file_path))?;

        info!(
            request_id = %request_id,
            prompt_tokens = metrics.prompt_tokens,
            completion_tokens = metrics.completion_tokens,
            latency_ms = metrics.e2e_latency_ms,
            cost = metrics.estimated_cost,
            "Read metrics from file"
        );

        // Generate TPM quote
        let quote = tpm_agent.generate_quote(&metrics);

        // Submit to coordinator
        let submission_result = if self.config.async_mode {
            // Async mode: return immediately
            match client.submit_metrics_async(request_id.clone(), metrics, quote).await {
                Ok(verification_id) => {
                    info!(
                        request_id = %request_id,
                        verification_id = %verification_id,
                        "Submitted (async - verification in background)"
                    );
                    Ok(())
                }
                Err(e) => {
                    Err(e)
                }
            }
        } else {
            // Sync mode: wait for verification result
            match client.submit_metrics(request_id.clone(), metrics, quote).await {
                Ok((verification_id, accepted)) => {
                    info!(
                        request_id = %request_id,
                        verification_id = %verification_id,
                        accepted,
                        "Submission result (sync)"
                    );

                    if !accepted {
                        return Err(anyhow::anyhow!("Metrics submission was rejected"));
                    }
                    Ok(())
                }
                Err(e) => {
                    Err(e)
                }
            }
        };

        // Move file based on result
        match submission_result {
            Ok(()) => {
                self.move_to_sent(file_path, &request_id)?;
                info!(request_id = %request_id, "File moved to sent");
            }
            Err(e) => {
                warn!(request_id = %request_id, error = %e, "Submission failed, moving to failed");
                self.move_to_failed(file_path, &request_id, &e.to_string())?;
            }
        }

        Ok(())
    }

    /// Read metrics from a JSONL file
    fn read_metrics_file(&self, file_path: &Path) -> Result<RequestMetrics> {
        let content = fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read file: {:?}", file_path))?;

        // Parse as Python RequestMetrics JSON
        let python_metrics: PythonRequestMetrics = serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse JSON from: {:?}", file_path))?;

        // Convert to Rust RequestMetrics (protobuf format)
        Ok(python_metrics.to_rust_metrics())
    }

    /// Get list of pending files, sorted by modification time (oldest first)
    fn get_pending_files(&self) -> Result<Vec<PathBuf>> {
        let mut files = Vec::new();

        let entries = fs::read_dir(&self.pending_dir)
            .with_context(|| format!("Failed to read pending directory: {:?}", self.pending_dir))?;

        for entry in entries {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("jsonl") {
                files.push(path);
            }
        }

        // Sort by modification time (oldest first for FIFO processing)
        files.sort_by_key(|path| {
            path.metadata()
                .and_then(|m| m.modified())
                .ok()
        });

        Ok(files)
    }

    /// Move file to sent directory
    fn move_to_sent(&self, file_path: &Path, request_id: &str) -> Result<()> {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let new_name = format!("{}_{}.jsonl", request_id, timestamp);
        let dst_path = self.sent_dir.join(new_name);

        fs::rename(file_path, &dst_path)
            .with_context(|| format!("Failed to move file to sent: {:?} -> {:?}", file_path, dst_path))?;

        Ok(())
    }

    /// Move file to failed directory
    fn move_to_failed(&self, file_path: &Path, request_id: &str, error_msg: &str) -> Result<()> {
        let timestamp = chrono::Local::now().format("%Y%m%d_%H%M%S");
        let new_name = format!("{}_{}.jsonl", request_id, timestamp);
        let dst_path = self.failed_dir.join(new_name);

        // Append error message to file
        let mut content = fs::read_to_string(file_path)
            .context("Failed to read file before moving to failed")?;

        content.push('\n');
        content.push_str(&serde_json::to_string(&serde_json::json!({
            "error": error_msg,
            "failed_at": timestamp.to_string(),
        }))?);

        fs::write(&dst_path, content)
            .with_context(|| format!("Failed to write to failed directory: {:?}", dst_path))?;

        fs::remove_file(file_path)
            .with_context(|| format!("Failed to remove original file: {:?}", file_path))?;

        Ok(())
    }
}

/// Python RequestMetrics format (matches Python dataclass)
#[derive(Debug, Serialize, Deserialize)]
struct PythonRequestMetrics {
    request_id: String,
    user_id: Option<String>,
    model_name: String,
    prompt_tokens: i32,
    completion_tokens: i32,
    cached_tokens: i32,
    e2e_latency: f64,
    time_to_first_token: f64,
    estimated_cost: f64,
    // Add more fields as needed
}

impl PythonRequestMetrics {
    /// Convert Python metrics to Rust RequestMetrics (protobuf format)
    fn to_rust_metrics(&self) -> RequestMetrics {
        RequestMetrics {
            prompt_tokens: self.prompt_tokens,
            completion_tokens: self.completion_tokens,
            cached_tokens: self.cached_tokens,
            e2e_latency_ms: (self.e2e_latency * 1000.0) as u64,
            time_to_first_token_ms: (self.time_to_first_token * 1000.0) as u64,
            estimated_cost: self.estimated_cost,
        }
    }
}
