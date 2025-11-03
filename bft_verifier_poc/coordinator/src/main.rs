use anyhow::Result;
use bft_coordinator::consensus::ConsensusManager;
use bft_coordinator::epoch::EpochManager;
use bft_coordinator::server::bft_verifier::coordinator_server::CoordinatorServer;
use bft_coordinator::server::CoordinatorService;
use bft_coordinator::vrf::VrfSelector;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use std::time::Duration;
use tonic::transport::Server;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CoordinatorConfig {
    /// Address to listen on
    listen_addr: String,

    /// VRF seed (hex-encoded 32 bytes)
    vrf_seed_hex: String,

    /// Committee size
    committee_size: usize,

    /// Quorum size (2f+1)
    quorum: usize,

    /// Vote collection timeout (seconds)
    vote_timeout_secs: u64,

    /// Epoch duration (seconds)
    epoch_duration_secs: u64,

    /// Verifier pool (list of verifier IDs)
    verifier_pool: Vec<String>,

    /// Verifier endpoints (verifier_id -> gRPC endpoint)
    verifier_endpoints: HashMap<String, String>,
}

impl Default for CoordinatorConfig {
    fn default() -> Self {
        let mut verifier_endpoints = HashMap::new();
        verifier_endpoints.insert(
            "verifier-1".to_string(),
            "http://localhost:50052".to_string(),
        );
        verifier_endpoints.insert(
            "verifier-2".to_string(),
            "http://localhost:50053".to_string(),
        );
        verifier_endpoints.insert(
            "verifier-3".to_string(),
            "http://localhost:50054".to_string(),
        );

        Self {
            listen_addr: "0.0.0.0:50051".to_string(),
            vrf_seed_hex: hex::encode([0u8; 32]),
            committee_size: 3,
            quorum: 2,
            vote_timeout_secs: 5,
            epoch_duration_secs: 30,
            verifier_pool: vec![
                "verifier-1".to_string(),
                "verifier-2".to_string(),
                "verifier-3".to_string(),
            ],
            verifier_endpoints,
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

    info!("Starting BFT Coordinator...");

    // Load configuration
    let config = load_config()?;
    info!("Configuration loaded");

    // Parse VRF seed
    let vrf_seed_bytes = hex::decode(&config.vrf_seed_hex)
        .map_err(|e| anyhow::anyhow!("Invalid VRF seed hex: {}", e))?;
    if vrf_seed_bytes.len() != 32 {
        return Err(anyhow::anyhow!("VRF seed must be 32 bytes"));
    }
    let mut vrf_seed = [0u8; 32];
    vrf_seed.copy_from_slice(&vrf_seed_bytes);

    // Create VRF selector
    let vrf_selector = VrfSelector::new(vrf_seed, config.committee_size);
    info!(
        committee_size = config.committee_size,
        "VRF selector initialized"
    );

    // Create consensus manager
    let consensus_manager = ConsensusManager::new(
        config.quorum,
        Duration::from_secs(config.vote_timeout_secs),
    );
    info!(
        quorum = config.quorum,
        vote_timeout_secs = config.vote_timeout_secs,
        "Consensus manager initialized"
    );

    // Create epoch manager
    let epoch_manager = EpochManager::new(Duration::from_secs(config.epoch_duration_secs));
    info!(
        epoch_duration_secs = config.epoch_duration_secs,
        current_epoch = epoch_manager.current_epoch(),
        "Epoch manager initialized"
    );

    // Log verifier pool
    info!(
        verifier_count = config.verifier_pool.len(),
        verifiers = ?config.verifier_pool,
        "Verifier pool loaded"
    );

    // Create coordinator service
    let service = CoordinatorService::new(
        vrf_selector,
        consensus_manager,
        epoch_manager,
        config.verifier_pool,
        config.verifier_endpoints,
    );

    // Parse listen address
    let addr: SocketAddr = config
        .listen_addr
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid listen address: {}", e))?;

    info!("Starting gRPC server on {}", addr);

    // Start server
    Server::builder()
        .add_service(CoordinatorServer::new(service))
        .serve(addr)
        .await
        .map_err(|e| {
            error!("Server error: {}", e);
            anyhow::anyhow!("Server failed: {}", e)
        })?;

    Ok(())
}

/// Load configuration from file or environment variables
fn load_config() -> Result<CoordinatorConfig> {
    // Try to load from file first
    let config_path =
        std::env::var("COORDINATOR_CONFIG").unwrap_or_else(|_| "coordinator_config.json".to_string());

    if std::path::Path::new(&config_path).exists() {
        info!("Loading config from file: {}", config_path);
        let config_str = fs::read_to_string(&config_path)?;
        let config: CoordinatorConfig = serde_json::from_str(&config_str)?;
        return Ok(config);
    }

    // Otherwise, load from environment variables
    info!("No config file found, using environment variables and defaults");
    let mut config = CoordinatorConfig::default();

    if let Ok(addr) = std::env::var("LISTEN_ADDR") {
        config.listen_addr = addr;
    }

    if let Ok(seed) = std::env::var("VRF_SEED") {
        config.vrf_seed_hex = seed;
    }

    if let Ok(size) = std::env::var("COMMITTEE_SIZE") {
        config.committee_size = size.parse()?;
    }

    if let Ok(quorum) = std::env::var("QUORUM") {
        config.quorum = quorum.parse()?;
    }

    if let Ok(timeout) = std::env::var("VOTE_TIMEOUT_SECS") {
        config.vote_timeout_secs = timeout.parse()?;
    }

    if let Ok(duration) = std::env::var("EPOCH_DURATION_SECS") {
        config.epoch_duration_secs = duration.parse()?;
    }

    if let Ok(pool_json) = std::env::var("VERIFIER_POOL_JSON") {
        config.verifier_pool = serde_json::from_str(&pool_json)?;
    }

    if let Ok(endpoints_json) = std::env::var("VERIFIER_ENDPOINTS_JSON") {
        config.verifier_endpoints = serde_json::from_str(&endpoints_json)?;
    }

    Ok(config)
}
