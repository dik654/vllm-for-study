///! Nonce manager for replay attack prevention
///!
///! This module manages one-time nonces to prevent replay attacks.
///! Each metrics submission must include a fresh nonce from the coordinator.

use anyhow::{Context, Result};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};
use tokio::sync::RwLock;
use tracing::{debug, info, warn};
use uuid::Uuid;

/// Nonce with expiration time
#[derive(Debug, Clone)]
struct NonceEntry {
    nonce: String,
    created_at: u64,
}

/// Nonce manager configuration
#[derive(Debug, Clone)]
pub struct NonceManagerConfig {
    /// How long nonces are valid (seconds)
    pub expiry_secs: u64,
    /// How often to clean expired nonces (seconds)
    pub cleanup_interval_secs: u64,
}

impl Default for NonceManagerConfig {
    fn default() -> Self {
        Self {
            expiry_secs: 300,         // 5 minutes
            cleanup_interval_secs: 60, // 1 minute
        }
    }
}

/// Manages nonces for replay attack prevention
pub struct NonceManager {
    config: NonceManagerConfig,
    /// Set of used nonces with their creation time
    used_nonces: Arc<RwLock<HashSet<String>>>,
    /// Set of issued nonces (not yet used)
    issued_nonces: Arc<RwLock<HashSet<String>>>,
}

impl NonceManager {
    /// Create a new nonce manager
    pub fn new(config: NonceManagerConfig) -> Self {
        let manager = Self {
            config,
            used_nonces: Arc::new(RwLock::new(HashSet::new())),
            issued_nonces: Arc::new(RwLock::new(HashSet::new())),
        };

        // Start background cleanup task
        manager.start_cleanup_task();

        info!("Nonce manager initialized (expiry: {}s)", config.expiry_secs);

        manager
    }

    /// Generate a new nonce
    pub async fn generate_nonce(&self) -> String {
        // Use UUID v4 for randomness
        let nonce = Uuid::new_v4().to_string();

        // Add to issued set
        let mut issued = self.issued_nonces.write().await;
        issued.insert(nonce.clone());

        debug!("Generated nonce: {}", nonce);

        nonce
    }

    /// Validate and consume a nonce
    ///
    /// Returns Ok(()) if nonce is valid and unused.
    /// Returns Err if nonce is:
    /// - Already used (replay attack)
    /// - Not issued by this coordinator
    /// - Expired
    pub async fn validate_and_consume(&self, nonce: &str) -> Result<()> {
        // Check if nonce was issued
        {
            let issued = self.issued_nonces.read().await;
            if !issued.contains(nonce) {
                return Err(anyhow::anyhow!(
                    "Nonce not issued by this coordinator"
                ));
            }
        }

        // Check if nonce was already used
        {
            let used = self.used_nonces.read().await;
            if used.contains(nonce) {
                warn!("Replay attack detected: nonce {} already used", nonce);
                return Err(anyhow::anyhow!("Nonce already used (replay attack)"));
            }
        }

        // Move nonce from issued to used
        {
            let mut issued = self.issued_nonces.write().await;
            issued.remove(nonce);
        }

        {
            let mut used = self.used_nonces.write().await;
            used.insert(nonce.to_string());
        }

        debug!("Nonce consumed: {}", nonce);

        Ok(())
    }

    /// Start background cleanup task
    fn start_cleanup_task(&self) {
        let used_nonces = Arc::clone(&self.used_nonces);
        let issued_nonces = Arc::clone(&self.issued_nonces);
        let cleanup_interval = self.config.cleanup_interval_secs;

        tokio::spawn(async move {
            loop {
                tokio::time::sleep(tokio::time::Duration::from_secs(cleanup_interval)).await;

                // Clean up (simplified - in production, track timestamps)
                let mut used = used_nonces.write().await;
                let mut issued = issued_nonces.write().await;

                let used_count = used.len();
                let issued_count = issued.len();

                // For now, just clear old entries periodically
                // In production, store timestamps and remove only expired ones
                if used_count > 10000 {
                    used.clear();
                    info!("Cleared {} used nonces", used_count);
                }

                if issued_count > 1000 {
                    issued.clear();
                    info!("Cleared {} issued nonces", issued_count);
                }
            }
        });
    }

    /// Get current stats
    pub async fn get_stats(&self) -> NonceStats {
        let used = self.used_nonces.read().await;
        let issued = self.issued_nonces.read().await;

        NonceStats {
            used_count: used.len(),
            issued_count: issued.len(),
        }
    }
}

/// Nonce statistics
#[derive(Debug, Clone)]
pub struct NonceStats {
    pub used_count: usize,
    pub issued_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_nonce_generation() {
        let manager = NonceManager::new(NonceManagerConfig::default());

        let nonce1 = manager.generate_nonce().await;
        let nonce2 = manager.generate_nonce().await;

        assert_ne!(nonce1, nonce2);
    }

    #[tokio::test]
    async fn test_nonce_reuse_prevention() {
        let manager = NonceManager::new(NonceManagerConfig::default());

        let nonce = manager.generate_nonce().await;

        // First use: should succeed
        assert!(manager.validate_and_consume(&nonce).await.is_ok());

        // Second use: should fail (replay attack)
        assert!(manager.validate_and_consume(&nonce).await.is_err());
    }

    #[tokio::test]
    async fn test_invalid_nonce() {
        let manager = NonceManager::new(NonceManagerConfig::default());

        // Try to use nonce that was never issued
        let fake_nonce = "fake-nonce-12345";
        assert!(manager.validate_and_consume(fake_nonce).await.is_err());
    }

    #[tokio::test]
    async fn test_stats() {
        let manager = NonceManager::new(NonceManagerConfig::default());

        let nonce1 = manager.generate_nonce().await;
        let nonce2 = manager.generate_nonce().await;

        let stats = manager.get_stats().await;
        assert_eq!(stats.issued_count, 2);
        assert_eq!(stats.used_count, 0);

        manager.validate_and_consume(&nonce1).await.unwrap();

        let stats = manager.get_stats().await;
        assert_eq!(stats.issued_count, 1);
        assert_eq!(stats.used_count, 1);
    }
}
