use rand::seq::SliceRandom;
use rand::{RngCore, SeedableRng};
use rand::rngs::StdRng;
use sha2::{Digest, Sha256};

/// VRF-based verifier selector
///
/// Provides deterministic, unpredictable, and fair verifier selection
/// based on VRF (Verifiable Random Function) principles.
pub struct VrfSelector {
    /// VRF seed (32 bytes)
    vrf_seed: [u8; 32],

    /// Committee size (number of verifiers to select)
    committee_size: usize,
}

impl VrfSelector {
    /// Create new VRF selector
    pub fn new(vrf_seed: [u8; 32], committee_size: usize) -> Self {
        Self {
            vrf_seed,
            committee_size,
        }
    }

    /// Select committee for given epoch
    ///
    /// Properties:
    /// - Deterministic: Same epoch → same committee
    /// - Unpredictable: Cannot predict future committees
    /// - Fair: All verifiers have equal probability
    /// - Verifiable: Anyone can verify selection
    pub fn select_committee(&self, epoch: u64, verifier_pool: &[String]) -> Vec<String> {
        if verifier_pool.is_empty() {
            return Vec::new();
        }

        if verifier_pool.len() <= self.committee_size {
            // If pool is smaller than committee size, return all
            return verifier_pool.to_vec();
        }

        // VRF: Hash(seed || epoch) → deterministic randomness
        let mut hasher = Sha256::new();
        hasher.update(&self.vrf_seed);
        hasher.update(&epoch.to_le_bytes());
        let hash = hasher.finalize();

        // Use hash as RNG seed
        let mut seed_bytes = [0u8; 32];
        seed_bytes.copy_from_slice(&hash[0..32]);
        let seed = u64::from_le_bytes(seed_bytes[0..8].try_into().unwrap());

        // Fisher-Yates shuffle with deterministic RNG
        let mut rng = StdRng::seed_from_u64(seed);
        let mut pool = verifier_pool.to_vec();
        pool.shuffle(&mut rng);

        // Take first committee_size verifiers
        pool.into_iter().take(self.committee_size).collect()
    }

    /// Get committee size
    pub fn committee_size(&self) -> usize {
        self.committee_size
    }

    /// Update VRF seed (for security rotation)
    pub fn rotate_seed(&mut self, new_seed: [u8; 32]) {
        self.vrf_seed = new_seed;
    }

    /// Generate random VRF seed
    pub fn generate_random_seed() -> [u8; 32] {
        let mut seed = [0u8; 32];
        rand::thread_rng().fill_bytes(&mut seed);
        seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_selection() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
            "v5".to_string(),
        ];

        // Same epoch should produce same committee
        let committee1 = selector.select_committee(1, &verifiers);
        let committee2 = selector.select_committee(1, &verifiers);

        assert_eq!(committee1, committee2);
        assert_eq!(committee1.len(), 3);
    }

    #[test]
    fn test_different_epochs_different_committees() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
            "v5".to_string(),
        ];

        let committee1 = selector.select_committee(1, &verifiers);
        let committee2 = selector.select_committee(2, &verifiers);

        // Different epochs should (usually) produce different committees
        assert_ne!(committee1, committee2);
    }

    #[test]
    fn test_different_seeds_different_committees() {
        let selector1 = VrfSelector::new([0u8; 32], 3);
        let selector2 = VrfSelector::new([1u8; 32], 3);
        let verifiers = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
            "v5".to_string(),
        ];

        let committee1 = selector1.select_committee(1, &verifiers);
        let committee2 = selector2.select_committee(1, &verifiers);

        // Different seeds should produce different committees
        assert_ne!(committee1, committee2);
    }

    #[test]
    fn test_empty_pool() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers: Vec<String> = vec![];

        let committee = selector.select_committee(1, &verifiers);
        assert_eq!(committee.len(), 0);
    }

    #[test]
    fn test_pool_smaller_than_committee() {
        let selector = VrfSelector::new([0u8; 32], 5);
        let verifiers = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];

        let committee = selector.select_committee(1, &verifiers);

        // Should return all verifiers when pool < committee_size
        assert_eq!(committee.len(), 3);
        assert_eq!(committee, verifiers);
    }

    #[test]
    fn test_pool_equal_to_committee() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec!["v1".to_string(), "v2".to_string(), "v3".to_string()];

        let committee = selector.select_committee(1, &verifiers);

        // Should return all verifiers
        assert_eq!(committee.len(), 3);
    }

    #[test]
    fn test_fairness_distribution() {
        // Test that over many epochs, each verifier is selected roughly equally
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
            "v5".to_string(),
        ];

        let mut selection_count = std::collections::HashMap::new();
        for v in &verifiers {
            selection_count.insert(v.clone(), 0);
        }

        // Run 1000 epochs
        for epoch in 0..1000 {
            let committee = selector.select_committee(epoch, &verifiers);
            for v in committee {
                *selection_count.get_mut(&v).unwrap() += 1;
            }
        }

        // Each verifier should be selected approximately 600 times (3/5 * 1000)
        // We allow 20% deviation (480-720 range)
        for (verifier, count) in selection_count {
            println!("{}: {} selections", verifier, count);
            assert!(
                count >= 480 && count <= 720,
                "Verifier {} selected {} times (expected 480-720)",
                verifier,
                count
            );
        }
    }

    #[test]
    fn test_committee_no_duplicates() {
        let selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
            "v5".to_string(),
        ];

        for epoch in 0..100 {
            let committee = selector.select_committee(epoch, &verifiers);

            // Check no duplicates
            let mut seen = std::collections::HashSet::new();
            for v in &committee {
                assert!(seen.insert(v.clone()), "Duplicate verifier in committee: {}", v);
            }
        }
    }

    #[test]
    fn test_seed_rotation() {
        let mut selector = VrfSelector::new([0u8; 32], 3);
        let verifiers = vec![
            "v1".to_string(),
            "v2".to_string(),
            "v3".to_string(),
            "v4".to_string(),
        ];

        let committee1 = selector.select_committee(1, &verifiers);

        // Rotate seed
        selector.rotate_seed([1u8; 32]);

        let committee2 = selector.select_committee(1, &verifiers);

        // Same epoch but different seed should give different committee
        assert_ne!(committee1, committee2);
    }

    #[test]
    fn test_generate_random_seed() {
        let seed1 = VrfSelector::generate_random_seed();
        let seed2 = VrfSelector::generate_random_seed();

        // Should generate different seeds
        assert_ne!(seed1, seed2);
        assert_eq!(seed1.len(), 32);
        assert_eq!(seed2.len(), 32);
    }

    #[test]
    fn test_committee_size_getter() {
        let selector = VrfSelector::new([0u8; 32], 5);
        assert_eq!(selector.committee_size(), 5);
    }
}
