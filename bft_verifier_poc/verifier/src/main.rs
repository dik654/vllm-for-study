use anyhow::Result;
use bft_common::{generate_keypair, signing_key_from_bytes, verifying_key_from_bytes};
use bft_verifier::server::bft_verifier::verifier_server::VerifierServer;
use bft_verifier::server::VerifierService;
use bft_verifier::tpm::SimulatedTpmVerifier;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::net::SocketAddr;
use tonic::transport::Server;
use tracing::{error, info};
use tracing_subscriber;

#[derive(Debug, Clone, Serialize, Deserialize)]
struct VerifierConfig {
    /// This verifier's ID
    verifier_id: String,

    /// Address to listen on (e.g., "0.0.0.0:50052")
    listen_addr: String,

    /// Signing key (hex-encoded 32 bytes)
    /// If empty, generate a new one
    signing_key_hex: String,

    /// Agent public keys (agent_id -> hex-encoded 32-byte public key)
    agent_public_keys: HashMap<String, String>,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            verifier_id: "verifier-1".to_string(),
            listen_addr: "0.0.0.0:50052".to_string(),
            signing_key_hex: String::new(),
            agent_public_keys: HashMap::new(),
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

    info!("Starting BFT Verifier...");

    // Load configuration
    let config = load_config()?;
    info!("Loaded configuration for verifier: {}", config.verifier_id);

    // Load or generate signing key
    let signing_key = if config.signing_key_hex.is_empty() {
        info!("No signing key provided, generating new one...");
        let (sk, _) = generate_keypair();
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

    // Load agent public keys
    let mut agent_keys = HashMap::new();
    for (agent_id, pubkey_hex) in &config.agent_public_keys {
        let key_bytes = hex::decode(pubkey_hex)
            .map_err(|e| anyhow::anyhow!("Invalid public key hex for {}: {}", agent_id, e))?;
        if key_bytes.len() != 32 {
            return Err(anyhow::anyhow!("Public key for {} must be 32 bytes", agent_id));
        }
        let mut key_array = [0u8; 32];
        key_array.copy_from_slice(&key_bytes);
        let verifying_key = verifying_key_from_bytes(&key_array)?;
        agent_keys.insert(agent_id.clone(), verifying_key);
        info!("Loaded public key for agent: {}", agent_id);
    }

    // Create TPM verifier
    let tpm_verifier = SimulatedTpmVerifier::new(agent_keys);
    info!("TPM verifier initialized with {} agents", tpm_verifier.agent_count());

    // Create gRPC service
    let service = VerifierService::new(
        config.verifier_id.clone(),
        tpm_verifier,
        signing_key,
    );

    // Parse listen address
    let addr: SocketAddr = config
        .listen_addr
        .parse()
        .map_err(|e| anyhow::anyhow!("Invalid listen address: {}", e))?;

    info!("Starting gRPC server on {}", addr);

    // Start server
    Server::builder()
        .add_service(VerifierServer::new(service))
        .serve(addr)
        .await
        .map_err(|e| {
            error!("Server error: {}", e);
            anyhow::anyhow!("Server failed: {}", e)
        })?;

    Ok(())
}

/// Load configuration from file or environment variables
fn load_config() -> Result<VerifierConfig> {
    // Try to load from file first
    let config_path = std::env::var("VERIFIER_CONFIG").unwrap_or_else(|_| "verifier_config.json".to_string());

    if std::path::Path::new(&config_path).exists() {
        info!("Loading config from file: {}", config_path);
        let config_str = fs::read_to_string(&config_path)?;
        let config: VerifierConfig = serde_json::from_str(&config_str)?;
        return Ok(config);
    }

    // Otherwise, load from environment variables
    info!("No config file found, using environment variables");
    let mut config = VerifierConfig::default();

    if let Ok(id) = std::env::var("VERIFIER_ID") {
        config.verifier_id = id;
    }

    if let Ok(addr) = std::env::var("LISTEN_ADDR") {
        config.listen_addr = addr;
    }

    if let Ok(key) = std::env::var("SIGNING_KEY") {
        config.signing_key_hex = key;
    }

    // Load agent keys from AGENT_KEYS_JSON env var
    if let Ok(keys_json) = std::env::var("AGENT_KEYS_JSON") {
        config.agent_public_keys = serde_json::from_str(&keys_json)?;
    }

    Ok(config)
}
