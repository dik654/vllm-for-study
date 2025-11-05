///! Real TPM 2.0 agent using hardware TPM chip
///!
///! This module provides hardware-based TPM 2.0 quote generation.
///! Requires actual TPM 2.0 hardware or simulator.
///!
///! Dependencies:
///! - tss-esapi: TPM 2.0 Software Stack
///! - libtss2-esys-dev: System library
///!
///! Usage:
///! ```bash
///! cargo build --features tpm-hardware
///! ```

#[cfg(feature = "tpm-hardware")]
use anyhow::{Context, Result};
use bft_common::{current_timestamp, RequestMetrics};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use tracing::{debug, error, info, warn};

// Protobuf TpmQuote structure
#[derive(Debug, Clone)]
pub struct TpmQuote {
    pub pcr_values: HashMap<u32, Vec<u8>>,
    pub quote_signature: Vec<u8>,
    pub nonce: Vec<u8>,
}

#[cfg(feature = "tpm-hardware")]
use tss_esapi::{
    abstraction::cipher::Cipher,
    handles::KeyHandle,
    interface_types::{
        algorithm::HashingAlgorithm,
        resource_handles::Hierarchy,
        session_handles::AuthSession,
    },
    structures::{
        Auth, Data, Digest as TpmDigest, HashScheme, PcrSelectionList, PcrSlot,
        SignatureScheme, SymmetricDefinition,
    },
    Context as TpmContext, Tcti, TctiNameConf,
};

/// Real TPM 2.0 agent
#[cfg(feature = "tpm-hardware")]
pub struct RealTpmAgent {
    tpm_context: TpmContext,
    signing_key_handle: KeyHandle,
    pcr_index: PcrSlot,
}

#[cfg(feature = "tpm-hardware")]
impl RealTpmAgent {
    /// Create a new TPM agent
    ///
    /// # Arguments
    /// * `tcti_config` - TCTI configuration (e.g., "device:/dev/tpmrm0")
    /// * `key_handle_value` - Persistent key handle (e.g., 0x81010001)
    /// * `pcr_index` - PCR index for metrics (default: 10)
    pub fn new(
        tcti_config: &str,
        key_handle_value: u32,
        pcr_index: u32,
    ) -> Result<Self> {
        info!("Initializing real TPM 2.0 agent");
        info!("  TCTI: {}", tcti_config);
        info!("  Key handle: 0x{:08x}", key_handle_value);
        info!("  PCR index: {}", pcr_index);

        // Parse TCTI configuration
        let tcti_conf = TctiNameConf::from_str(tcti_config)
            .context("Failed to parse TCTI config")?;

        // Initialize TCTI
        let tcti = Tcti::from_tcti_name_conf(tcti_conf)
            .context("Failed to initialize TCTI")?;

        // Create TPM context
        let tpm_context = TpmContext::new(tcti)
            .context("Failed to create TPM context")?;

        // Convert key handle
        let signing_key_handle = KeyHandle::from(key_handle_value);

        // Convert PCR index
        let pcr_slot = match pcr_index {
            10 => PcrSlot::Slot10,
            16 => PcrSlot::Slot16,
            _ => {
                return Err(anyhow::anyhow!(
                    "Unsupported PCR index: {}. Use 10 or 16",
                    pcr_index
                ))
            }
        };

        info!("TPM agent initialized successfully");

        Ok(Self {
            tpm_context,
            signing_key_handle,
            pcr_index: pcr_slot,
        })
    }

    /// Generate TPM quote for metrics
    pub fn generate_quote(&mut self, metrics: &RequestMetrics) -> Result<TpmQuote> {
        debug!("Generating TPM quote for metrics");

        // 1. Hash metrics
        let metrics_hash = self.hash_metrics(metrics)?;
        debug!("Metrics hash: {}", hex::encode(&metrics_hash));

        // 2. Extend PCR with metrics hash
        self.extend_pcr(&metrics_hash)
            .context("Failed to extend PCR")?;

        // 3. Read PCR value
        let pcr_values = self.read_pcr()
            .context("Failed to read PCR")?;

        // 4. Create TPM quote
        let quote_signature = self
            .create_quote(&metrics_hash)
            .context("Failed to create quote")?;

        info!("TPM quote generated successfully");

        Ok(TpmQuote {
            pcr_values,
            quote_signature,
            nonce: metrics_hash,
        })
    }

    /// Hash metrics to bytes
    fn hash_metrics(&self, metrics: &RequestMetrics) -> Result<Vec<u8>> {
        let mut hasher = Sha256::new();

        // Hash all fields in deterministic order
        hasher.update(metrics.sequence.to_le_bytes());
        hasher.update(&metrics.prev_hash);
        hasher.update(metrics.prompt_tokens.to_le_bytes());
        hasher.update(metrics.completion_tokens.to_le_bytes());
        hasher.update(metrics.cached_tokens.to_le_bytes());
        hasher.update(metrics.e2e_latency_ms.to_le_bytes());
        hasher.update(metrics.time_to_first_token_ms.to_le_bytes());
        hasher.update(metrics.estimated_cost.to_le_bytes());

        Ok(hasher.finalize().to_vec())
    }

    /// Extend PCR with hash
    fn extend_pcr(&mut self, hash: &[u8]) -> Result<()> {
        debug!("Extending PCR {:?} with hash", self.pcr_index);

        // Create digest from hash
        let digest_values = vec![TpmDigest::try_from(hash.to_vec())
            .context("Failed to create digest")?];

        // Extend PCR
        self.tpm_context
            .execute_without_session(|ctx| {
                ctx.pcr_extend(self.pcr_index.into(), digest_values)
            })
            .context("PCR extend failed")?;

        debug!("PCR extended successfully");

        Ok(())
    }

    /// Read PCR value
    fn read_pcr(&mut self) -> Result<HashMap<u32, Vec<u8>>> {
        debug!("Reading PCR {:?}", self.pcr_index);

        // Build PCR selection list
        let pcr_selection_list = PcrSelectionList::builder()
            .with_selection(HashingAlgorithm::Sha256, &[self.pcr_index])
            .build()
            .context("Failed to build PCR selection")?;

        // Read PCR
        let (_update_counter, pcr_selections, pcr_data) = self
            .tpm_context
            .pcr_read(pcr_selection_list)
            .context("Failed to read PCR")?;

        // Extract PCR value
        let mut pcr_values = HashMap::new();

        if let Some(pcr_bank) = pcr_data.pcr_bank(HashingAlgorithm::Sha256) {
            let pcr_index_num = match self.pcr_index {
                PcrSlot::Slot10 => 10,
                PcrSlot::Slot16 => 16,
                _ => 10,
            };

            if let Some(digest) = pcr_bank.get(self.pcr_index) {
                pcr_values.insert(pcr_index_num, digest.as_bytes().to_vec());
            }
        }

        debug!("PCR read successfully");

        Ok(pcr_values)
    }

    /// Create TPM quote
    fn create_quote(&mut self, qualifying_data: &[u8]) -> Result<Vec<u8>> {
        debug!("Creating TPM quote");

        // Build PCR selection list
        let pcr_selection_list = PcrSelectionList::builder()
            .with_selection(HashingAlgorithm::Sha256, &[self.pcr_index])
            .build()
            .context("Failed to build PCR selection")?;

        // Qualifying data
        let qualifying_data = Data::try_from(qualifying_data.to_vec())
            .context("Failed to create qualifying data")?;

        // Signature scheme
        let scheme = SignatureScheme::RsaSsa {
            hash_scheme: HashScheme::new(HashingAlgorithm::Sha256),
        };

        // Create quote
        let (attest, signature) = self
            .tpm_context
            .execute_without_session(|ctx| {
                ctx.quote(
                    self.signing_key_handle,
                    qualifying_data.clone(),
                    scheme,
                    pcr_selection_list.clone(),
                )
            })
            .context("Quote creation failed")?;

        debug!("TPM quote created successfully");

        // Return signature bytes
        Ok(signature.signature().to_vec())
    }

    /// Get public key for verification
    pub fn get_public_key(&mut self) -> Result<Vec<u8>> {
        let (public, _, _) = self
            .tpm_context
            .read_public(self.signing_key_handle.into())
            .context("Failed to read public key")?;

        // Extract public key bytes
        let public_bytes = public.try_into().context("Failed to convert public key")?;

        Ok(public_bytes)
    }
}

#[cfg(not(feature = "tpm-hardware"))]
pub struct RealTpmAgent;

#[cfg(not(feature = "tpm-hardware"))]
impl RealTpmAgent {
    pub fn new(_tcti_config: &str, _key_handle: u32, _pcr_index: u32) -> Result<Self> {
        Err(anyhow::anyhow!(
            "TPM hardware support not enabled. Build with --features tpm-hardware"
        ))
    }

    pub fn generate_quote(&mut self, _metrics: &RequestMetrics) -> Result<TpmQuote> {
        Err(anyhow::anyhow!(
            "TPM hardware support not enabled"
        ))
    }

    pub fn get_public_key(&mut self) -> Result<Vec<u8>> {
        Err(anyhow::anyhow!(
            "TPM hardware support not enabled"
        ))
    }
}

#[cfg(all(test, feature = "tpm-hardware"))]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires actual TPM hardware
    fn test_tpm_initialization() {
        let agent = RealTpmAgent::new("device:/dev/tpmrm0", 0x81010001, 10);
        assert!(agent.is_ok());
    }

    #[test]
    #[ignore] // Requires actual TPM hardware
    fn test_tpm_quote_generation() {
        let mut agent = RealTpmAgent::new("device:/dev/tpmrm0", 0x81010001, 10).unwrap();

        let metrics = RequestMetrics {
            prompt_tokens: 100,
            completion_tokens: 50,
            cached_tokens: 0,
            e2e_latency_ms: 1000,
            time_to_first_token_ms: 100,
            estimated_cost: 0.01,
            sequence: 1,
            prev_hash: vec![0u8; 32],
            current_hash: vec![],
        };

        let quote = agent.generate_quote(&metrics);
        assert!(quote.is_ok());

        let quote = quote.unwrap();
        assert!(!quote.pcr_values.is_empty());
        assert!(!quote.quote_signature.is_empty());
    }

    #[test]
    #[ignore] // Requires actual TPM hardware
    fn test_get_public_key() {
        let mut agent = RealTpmAgent::new("device:/dev/tpmrm0", 0x81010001, 10).unwrap();

        let public_key = agent.get_public_key();
        assert!(public_key.is_ok());

        let key = public_key.unwrap();
        assert!(!key.is_empty());
        println!("Public key: {}", hex::encode(&key));
    }
}
