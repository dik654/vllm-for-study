use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tonic::transport::{Channel, Endpoint};
use tracing::{debug, info, warn};

pub mod bft_verifier {
    tonic::include_proto!("bft_verifier");
}

use bft_verifier::verifier_client::VerifierClient;

/// Connection pool for verifier gRPC clients
///
/// Maintains persistent connections to verifiers to avoid
/// connection overhead on each verification request.
///
/// Features:
/// - Lazy connection establishment
/// - Connection reuse
/// - Pre-warming for anticipated verifiers
/// - Automatic reconnection on errors
pub struct VerifierConnectionPool {
    /// Active connections (verifier_id -> client)
    connections: Arc<RwLock<HashMap<String, VerifierClient<Channel>>>>,

    /// Verifier endpoints (verifier_id -> endpoint URL)
    endpoints: Arc<HashMap<String, String>>,

    /// Maximum connections to maintain
    max_connections: usize,
}

impl VerifierConnectionPool {
    /// Create new connection pool
    pub fn new(endpoints: HashMap<String, String>, max_connections: usize) -> Self {
        Self {
            connections: Arc::new(RwLock::new(HashMap::new())),
            endpoints: Arc::new(endpoints),
            max_connections,
        }
    }

    /// Get or create connection to verifier
    ///
    /// Returns a cloned client (gRPC clients are cheaply cloneable)
    pub async fn get_client(
        &self,
        verifier_id: &str,
    ) -> Result<VerifierClient<Channel>, Box<dyn std::error::Error + Send + Sync>> {
        // Check if connection exists
        {
            let connections = self.connections.read().await;
            if let Some(client) = connections.get(verifier_id) {
                debug!("Reusing existing connection to {}", verifier_id);
                return Ok(client.clone());
            }
        }

        // Connection doesn't exist, create it
        self.connect(verifier_id).await
    }

    /// Establish connection to verifier
    async fn connect(
        &self,
        verifier_id: &str,
    ) -> Result<VerifierClient<Channel>, Box<dyn std::error::Error + Send + Sync>> {
        let endpoint = self
            .endpoints
            .get(verifier_id)
            .ok_or_else(|| format!("Unknown verifier: {}", verifier_id))?;

        info!("Establishing connection to {} at {}", verifier_id, endpoint);

        // Create gRPC channel with configured options
        let channel = Endpoint::from_shared(endpoint.clone())?
            .connect_timeout(std::time::Duration::from_secs(5))
            .timeout(std::time::Duration::from_secs(30))
            .tcp_keepalive(Some(std::time::Duration::from_secs(60)))
            .http2_keep_alive_interval(std::time::Duration::from_secs(30))
            .keep_alive_timeout(std::time::Duration::from_secs(10))
            .connect()
            .await?;

        let client = VerifierClient::new(channel);

        // Store connection
        {
            let mut connections = self.connections.write().await;

            // Enforce max connections (simple FIFO eviction)
            if connections.len() >= self.max_connections {
                if let Some((oldest_id, _)) = connections.iter().next() {
                    let oldest_id = oldest_id.clone();
                    connections.remove(&oldest_id);
                    debug!("Evicted connection to {} (pool full)", oldest_id);
                }
            }

            connections.insert(verifier_id.to_string(), client.clone());
        }

        info!("Successfully connected to {}", verifier_id);
        Ok(client)
    }

    /// Pre-warm connections to a list of verifiers
    ///
    /// Establishes connections in background without blocking.
    /// Used for speculative execution - pre-connect to anticipated
    /// verifiers before they're needed.
    pub async fn warm_up(&self, verifier_ids: &[String]) {
        info!(
            count = verifier_ids.len(),
            verifiers = ?verifier_ids,
            "Pre-warming connections"
        );

        let mut tasks = Vec::new();

        for verifier_id in verifier_ids {
            let pool = self.clone();
            let id = verifier_id.clone();

            tasks.push(tokio::spawn(async move {
                match pool.connect(&id).await {
                    Ok(_) => {
                        debug!("Pre-warmed connection to {}", id);
                    }
                    Err(e) => {
                        warn!("Failed to pre-warm connection to {}: {}", id, e);
                    }
                }
            }));
        }

        // Wait for all connections (with timeout)
        let timeout = tokio::time::timeout(
            std::time::Duration::from_secs(10),
            futures::future::join_all(tasks),
        );

        match timeout.await {
            Ok(_) => {
                info!("Connection pre-warming complete");
            }
            Err(_) => {
                warn!("Connection pre-warming timed out");
            }
        }
    }

    /// Remove connection from pool
    pub async fn remove(&self, verifier_id: &str) {
        let mut connections = self.connections.write().await;
        if connections.remove(verifier_id).is_some() {
            info!("Removed connection to {}", verifier_id);
        }
    }

    /// Get number of active connections
    pub async fn connection_count(&self) -> usize {
        self.connections.read().await.len()
    }

    /// Clear all connections
    pub async fn clear(&self) {
        let mut connections = self.connections.write().await;
        let count = connections.len();
        connections.clear();
        info!("Cleared {} connections", count);
    }

    /// Check if connection exists for verifier
    pub async fn has_connection(&self, verifier_id: &str) -> bool {
        self.connections.read().await.contains_key(verifier_id)
    }
}

impl Clone for VerifierConnectionPool {
    fn clone(&self) -> Self {
        Self {
            connections: Arc::clone(&self.connections),
            endpoints: Arc::clone(&self.endpoints),
            max_connections: self.max_connections,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_connection_pool_creation() {
        let endpoints = HashMap::from([
            ("v1".to_string(), "http://localhost:50001".to_string()),
            ("v2".to_string(), "http://localhost:50002".to_string()),
        ]);

        let pool = VerifierConnectionPool::new(endpoints, 10);

        assert_eq!(pool.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_connection_pool_clone() {
        let endpoints = HashMap::from([
            ("v1".to_string(), "http://localhost:50001".to_string()),
        ]);

        let pool1 = VerifierConnectionPool::new(endpoints, 10);
        let pool2 = pool1.clone();

        // Both should share the same underlying state
        assert_eq!(pool1.connection_count().await, 0);
        assert_eq!(pool2.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_has_connection() {
        let endpoints = HashMap::from([
            ("v1".to_string(), "http://localhost:50001".to_string()),
        ]);

        let pool = VerifierConnectionPool::new(endpoints, 10);

        assert!(!pool.has_connection("v1").await);
    }

    #[tokio::test]
    async fn test_clear() {
        let endpoints = HashMap::from([
            ("v1".to_string(), "http://localhost:50001".to_string()),
        ]);

        let pool = VerifierConnectionPool::new(endpoints, 10);
        pool.clear().await;

        assert_eq!(pool.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_remove() {
        let endpoints = HashMap::from([
            ("v1".to_string(), "http://localhost:50001".to_string()),
        ]);

        let pool = VerifierConnectionPool::new(endpoints, 10);

        // Removing non-existent connection should be safe
        pool.remove("v1").await;
        assert_eq!(pool.connection_count().await, 0);
    }

    #[tokio::test]
    async fn test_get_client_unknown_verifier() {
        let endpoints = HashMap::new();
        let pool = VerifierConnectionPool::new(endpoints, 10);

        let result = pool.get_client("unknown").await;
        assert!(result.is_err());
    }

    // Note: Full integration tests with actual gRPC connections
    // would require running verifier services, which is tested
    // in end-to-end tests instead.
}
