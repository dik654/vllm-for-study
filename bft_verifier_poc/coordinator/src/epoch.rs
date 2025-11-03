use std::time::{Duration, Instant};

/// Epoch manager for committee rotation
///
/// Manages time-based epochs for periodic verifier committee rotation.
/// Epochs are sequential time windows of fixed duration.
pub struct EpochManager {
    /// Duration of each epoch
    epoch_duration: Duration,

    /// Start time (reference point for epoch 0)
    start_time: Instant,
}

impl EpochManager {
    /// Create new epoch manager
    ///
    /// # Arguments
    /// * `epoch_duration` - Duration of each epoch (e.g., 30 seconds)
    pub fn new(epoch_duration: Duration) -> Self {
        Self {
            epoch_duration,
            start_time: Instant::now(),
        }
    }

    /// Create epoch manager with custom start time
    pub fn with_start_time(epoch_duration: Duration, start_time: Instant) -> Self {
        Self {
            epoch_duration,
            start_time,
        }
    }

    /// Get current epoch number
    ///
    /// Epoch 0 starts at `start_time`, epoch 1 at `start_time + epoch_duration`, etc.
    pub fn current_epoch(&self) -> u64 {
        let elapsed = self.start_time.elapsed();
        elapsed.as_secs() / self.epoch_duration.as_secs()
    }

    /// Get time remaining until next epoch
    pub fn time_until_next_epoch(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        let current_epoch_num = self.current_epoch();
        let next_epoch_start_secs = (current_epoch_num + 1) * self.epoch_duration.as_secs();
        let next_epoch_start = Duration::from_secs(next_epoch_start_secs);

        next_epoch_start.saturating_sub(elapsed)
    }

    /// Get time elapsed in current epoch
    pub fn time_in_current_epoch(&self) -> Duration {
        let elapsed = self.start_time.elapsed();
        let current_epoch_num = self.current_epoch();
        let current_epoch_start_secs = current_epoch_num * self.epoch_duration.as_secs();
        let current_epoch_start = Duration::from_secs(current_epoch_start_secs);

        elapsed.saturating_sub(current_epoch_start)
    }

    /// Get epoch duration
    pub fn epoch_duration(&self) -> Duration {
        self.epoch_duration
    }

    /// Check if we're in a specific epoch
    pub fn is_epoch(&self, epoch: u64) -> bool {
        self.current_epoch() == epoch
    }

    /// Wait for next epoch
    pub async fn wait_for_next_epoch(&self) {
        let remaining = self.time_until_next_epoch();
        tokio::time::sleep(remaining).await;
    }

    /// Get start time of a specific epoch
    pub fn epoch_start_time(&self, epoch: u64) -> Instant {
        self.start_time + Duration::from_secs(epoch * self.epoch_duration.as_secs())
    }

    /// Get end time of a specific epoch
    pub fn epoch_end_time(&self, epoch: u64) -> Instant {
        self.start_time + Duration::from_secs((epoch + 1) * self.epoch_duration.as_secs())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::sleep;

    #[test]
    fn test_current_epoch() {
        let manager = EpochManager::new(Duration::from_secs(10));

        // Should start at epoch 0
        assert_eq!(manager.current_epoch(), 0);
    }

    #[test]
    fn test_epoch_progression() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_millis(100), start);

        // Immediately should be epoch 0
        assert_eq!(manager.current_epoch(), 0);

        // After 150ms should be epoch 1
        std::thread::sleep(Duration::from_millis(150));
        assert_eq!(manager.current_epoch(), 1);

        // After 250ms total should be epoch 2
        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(manager.current_epoch(), 2);
    }

    #[test]
    fn test_time_until_next_epoch() {
        let manager = EpochManager::new(Duration::from_secs(10));
        let remaining = manager.time_until_next_epoch();

        // Should be less than or equal to 10 seconds
        assert!(remaining <= Duration::from_secs(10));

        // Should be positive
        assert!(remaining > Duration::from_secs(0));
    }

    #[test]
    fn test_time_in_current_epoch() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_secs(10), start);

        std::thread::sleep(Duration::from_millis(500));

        let elapsed = manager.time_in_current_epoch();

        // Should be around 500ms
        assert!(elapsed >= Duration::from_millis(400));
        assert!(elapsed <= Duration::from_millis(600));
    }

    #[test]
    fn test_epoch_duration_getter() {
        let manager = EpochManager::new(Duration::from_secs(30));
        assert_eq!(manager.epoch_duration(), Duration::from_secs(30));
    }

    #[test]
    fn test_is_epoch() {
        let manager = EpochManager::new(Duration::from_secs(10));

        assert!(manager.is_epoch(0));
        assert!(!manager.is_epoch(1));
        assert!(!manager.is_epoch(100));
    }

    #[tokio::test]
    async fn test_wait_for_next_epoch() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_millis(100), start);

        assert_eq!(manager.current_epoch(), 0);

        manager.wait_for_next_epoch().await;

        // Should now be in epoch 1
        assert_eq!(manager.current_epoch(), 1);
    }

    #[test]
    fn test_epoch_start_time() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_secs(10), start);

        let epoch0_start = manager.epoch_start_time(0);
        let epoch1_start = manager.epoch_start_time(1);
        let epoch2_start = manager.epoch_start_time(2);

        assert_eq!(epoch0_start, start);
        assert_eq!(epoch1_start, start + Duration::from_secs(10));
        assert_eq!(epoch2_start, start + Duration::from_secs(20));
    }

    #[test]
    fn test_epoch_end_time() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_secs(10), start);

        let epoch0_end = manager.epoch_end_time(0);
        let epoch1_end = manager.epoch_end_time(1);

        assert_eq!(epoch0_end, start + Duration::from_secs(10));
        assert_eq!(epoch1_end, start + Duration::from_secs(20));
    }

    #[test]
    fn test_epoch_boundaries() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_millis(100), start);

        // Right at start should be epoch 0
        assert_eq!(manager.current_epoch(), 0);

        // Just before epoch 1
        std::thread::sleep(Duration::from_millis(90));
        assert_eq!(manager.current_epoch(), 0);

        // Just after epoch 1 starts
        std::thread::sleep(Duration::from_millis(20));
        assert_eq!(manager.current_epoch(), 1);
    }

    #[test]
    fn test_multiple_epochs() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_millis(50), start);

        for expected_epoch in 0..5 {
            assert_eq!(manager.current_epoch(), expected_epoch);
            std::thread::sleep(Duration::from_millis(60));
        }
    }

    #[tokio::test]
    async fn test_wait_respects_remaining_time() {
        let start = Instant::now();
        let manager = EpochManager::with_start_time(Duration::from_millis(100), start);

        // Sleep partway through epoch 0
        sleep(Duration::from_millis(30)).await;

        let before_wait = Instant::now();
        manager.wait_for_next_epoch().await;
        let after_wait = Instant::now();

        let wait_duration = after_wait - before_wait;

        // Should have waited approximately 70ms (100 - 30)
        assert!(wait_duration >= Duration::from_millis(60));
        assert!(wait_duration <= Duration::from_millis(90));
    }
}
