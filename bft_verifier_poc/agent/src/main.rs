use anyhow::Result;
use bft_agent::client::CoordinatorClient;
use bft_agent::file_monitor::{FileMonitor, FileMonitorConfig};
use bft_agent::metrics::MetricsGenerator;
use bft_agent::tpm::SimulatedTpmAgent;
use bft_common::{generate_keypair, signing_key_from_bytes, verifying_key_to_bytes};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;
use std::time::Duration;
use tokio::time::sleep;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct AgentConfig {
    /// This agent's ID
    agent_id: String,

    /// Coordinator endpoint
    coordinator_endpoint: String,

    /// Signing key (hex-encoded 32 bytes)
    /// If empty, generate a new one
    signing_key_hex: String,

    /// Submission interval (seconds)
    /// How often to generate and submit metrics
    submission_interval_secs: u64,

    /// Mode: "single", "continuous", or "file_monitor"
    mode: String,

    /// Number of requests to send (for single mode)
    request_count: usize,

    /// Async mode: if true, return immediately (don't wait for verification)
    /// Useful for simulating LLM workloads where verification happens in background
    async_mode: bool,

    /// File monitor base directory (for file_monitor mode)
    /// Default: /var/log/vllm/metrics
    file_monitor_dir: String,

    /// File monitor poll interval in seconds (for file_monitor mode)
    file_monitor_poll_secs: u64,

    /// File monitor batch size (for file_monitor mode)
    file_monitor_batch_size: usize,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            agent_id: "agent-1".to_string(),
            coordinator_endpoint: "http://localhost:50051".to_string(),
            signing_key_hex: String::new(),
            submission_interval_secs: 5,
            mode: "continuous".to_string(),
            request_count: 10,
            async_mode: false,  // Default: synchronous mode
            file_monitor_dir: "/var/log/vllm/metrics".to_string(),
            file_monitor_poll_secs: 1,
            file_monitor_batch_size: 10,
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    info!("Starting BFT Agent...");

    // Load configuration
    let config = load_config()?;
    info!("Configuration loaded for agent: {}", config.agent_id);

    // Load or generate signing key
    let signing_key = if config.signing_key_hex.is_empty() {
        info!("No signing key provided, generating new one...");
        let (sk, vk) = generate_keypair();
        let vk_bytes = verifying_key_to_bytes(&vk);
        info!(
            "Generated new keypair. Public key (hex): {}",
            hex::encode(vk_bytes)
        );
        info!("IMPORTANT: Save this public key and configure verifiers with it!");
        sk
    } else {
        info!("Loading signing key from config...");
        let key_bytes = hex::decode(&config.signing_key_hex)
            .map_err(|e| anyhow::anyhow!("Invalid signing key hex: {}", e))?;
        if key_bytes.len() != 32 {
            return Err(anyhow::anyhow!("Signing key must be 32 bytes"));
        }
        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        signing_key_from_bytes(&key_array)
    };

    // Create TPM agent
    let tpm_agent = SimulatedTpmAgent::new(signing_key);
    info!("TPM agent initialized");

    // Create metrics generator
    let mut metrics_gen = MetricsGenerator::new();
    info!("Metrics generator initialized");

    // Connect to coordinator
    info!("Connecting to coordinator at {}", config.coordinator_endpoint);
    let mut client = CoordinatorClient::connect(
        config.coordinator_endpoint.clone(),
        config.agent_id.clone(),
    )
    .await?;
    info!("Connected to coordinator");

    // Run based on mode
    info!("Verification mode: {}", if config.async_mode { "async" } else { "sync" });

    match config.mode.as_str() {
        "single" => {
            info!("Running in single mode ({} requests)", config.request_count);
            run_single_mode(
                &mut client,
                &tpm_agent,
                &mut metrics_gen,
                config.request_count,
                Duration::from_secs(config.submission_interval_secs),
                config.async_mode,
            )
            .await?;
        }
        "continuous" => {
            info!("Running in continuous mode (interval: {}s)", config.submission_interval_secs);
            run_continuous_mode(
                &mut client,
                &tpm_agent,
                &mut metrics_gen,
                Duration::from_secs(config.submission_interval_secs),
                config.async_mode,
            )
            .await?;
        }
        "file_monitor" => {
            info!("Running in file_monitor mode");
            info!("  Directory: {}", config.file_monitor_dir);
            info!("  Poll interval: {}s", config.file_monitor_poll_secs);
            info!("  Batch size: {}", config.file_monitor_batch_size);

            let monitor_config = FileMonitorConfig {
                base_dir: PathBuf::from(config.file_monitor_dir),
                poll_interval_secs: config.file_monitor_poll_secs,
                batch_size: config.file_monitor_batch_size,
                async_mode: config.async_mode,
            };

            let file_monitor = FileMonitor::new(monitor_config)?;
            file_monitor.run(&mut client, &tpm_agent).await?;
        }
        _ => {
            return Err(anyhow::anyhow!("Invalid mode: {}. Valid modes: single, continuous, file_monitor", config.mode));
        }
    }

    Ok(())
}

/// Run in single mode (send N requests and exit)
async fn run_single_mode(
    client: &mut CoordinatorClient,
    tpm_agent: &SimulatedTpmAgent,
    metrics_gen: &mut MetricsGenerator,
    count: usize,
    interval: Duration,
    async_mode: bool,
) -> Result<()> {
    for i in 0..count {
        let request_id = format!("req-{}-{}", client.agent_id(), i);

        // Generate metrics
        let metrics = metrics_gen.generate_realistic();
        info!(
            request_id = %request_id,
            prompt_tokens = metrics.prompt_tokens,
            completion_tokens = metrics.completion_tokens,
            latency_ms = metrics.e2e_latency_ms,
            cost = metrics.estimated_cost,
            "Generated metrics"
        );

        // Generate TPM quote
        let quote = tpm_agent.generate_quote(&metrics);

        // Submit to coordinator
        if async_mode {
            // Async mode: return immediately
            match client.submit_metrics_async(request_id.clone(), metrics, quote).await {
                Ok(verification_id) => {
                    info!(
                        request_id = %request_id,
                        verification_id = %verification_id,
                        "Submitted (async - verification in background)"
                    );
                }
                Err(e) => {
                    error!(request_id = %request_id, error = %e, "Submission failed");
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
                        error!("Metrics submission was rejected!");
                    }
                }
                Err(e) => {
                    error!(request_id = %request_id, error = %e, "Submission failed");
                }
            }
        }

        // Wait before next submission
        if i < count - 1 {
            sleep(interval).await;
        }
    }

    info!("Completed {} submissions", count);
    Ok(())
}

/// Run in continuous mode (send requests indefinitely)
async fn run_continuous_mode(
    client: &mut CoordinatorClient,
    tpm_agent: &SimulatedTpmAgent,
    metrics_gen: &mut MetricsGenerator,
    interval: Duration,
    async_mode: bool,
) -> Result<()> {
    let mut counter = 0;

    loop {
        let request_id = format!("req-{}-{}", client.agent_id(), counter);
        counter += 1;

        // Generate metrics
        let metrics = metrics_gen.generate_realistic();
        info!(
            request_id = %request_id,
            prompt_tokens = metrics.prompt_tokens,
            completion_tokens = metrics.completion_tokens,
            "Generated metrics"
        );

        // Generate TPM quote
        let quote = tpm_agent.generate_quote(&metrics);

        // Submit to coordinator
        if async_mode {
            // Async mode: return immediately
            match client.submit_metrics_async(request_id.clone(), metrics, quote).await {
                Ok(verification_id) => {
                    info!(
                        request_id = %request_id,
                        verification_id = %verification_id,
                        "Submitted (async - verification in background)"
                    );
                }
                Err(e) => {
                    error!(error = %e, "Submission failed");
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
                        "Submitted (sync)"
                    );

                    if !accepted {
                        error!("Submission rejected!");
                    }
                }
                Err(e) => {
                    error!(error = %e, "Submission failed");
                }
            }
        }

        sleep(interval).await;
    }
}

/// Load configuration from file or environment variables
fn load_config() -> Result<AgentConfig> {
    // Try to load from file first
    let config_path = std::env::var("AGENT_CONFIG").unwrap_or_else(|_| "agent_config.json".to_string());

    if std::path::Path::new(&config_path).exists() {
        info!("Loading config from file: {}", config_path);
        let config_str = fs::read_to_string(&config_path)?;
        let config: AgentConfig = serde_json::from_str(&config_str)?;
        return Ok(config);
    }

    // Otherwise, load from environment variables
    info!("No config file found, using environment variables and defaults");
    let mut config = AgentConfig::default();

    if let Ok(id) = std::env::var("AGENT_ID") {
        config.agent_id = id;
    }

    if let Ok(endpoint) = std::env::var("COORDINATOR_ENDPOINT") {
        config.coordinator_endpoint = endpoint;
    }

    if let Ok(key) = std::env::var("SIGNING_KEY") {
        config.signing_key_hex = key;
    }

    if let Ok(interval) = std::env::var("SUBMISSION_INTERVAL_SECS") {
        config.submission_interval_secs = interval.parse()?;
    }

    if let Ok(mode) = std::env::var("MODE") {
        config.mode = mode;
    }

    if let Ok(count) = std::env::var("REQUEST_COUNT") {
        config.request_count = count.parse()?;
    }

    if let Ok(async_mode) = std::env::var("ASYNC_MODE") {
        config.async_mode = async_mode.parse().unwrap_or(false);
    }

    if let Ok(dir) = std::env::var("FILE_MONITOR_DIR") {
        config.file_monitor_dir = dir;
    }

    if let Ok(poll_secs) = std::env::var("FILE_MONITOR_POLL_SECS") {
        config.file_monitor_poll_secs = poll_secs.parse().unwrap_or(1);
    }

    if let Ok(batch_size) = std::env::var("FILE_MONITOR_BATCH_SIZE") {
        config.file_monitor_batch_size = batch_size.parse().unwrap_or(10);
    }

    Ok(config)
}
