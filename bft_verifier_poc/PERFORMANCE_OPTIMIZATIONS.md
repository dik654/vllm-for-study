# BFT Verifier PoC - Performance Optimization Guide

## Overview

This document describes performance optimizations for the BFT consensus system to minimize impact on LLM request latency.

---

## Current Performance Characteristics

### Baseline Design

**Current Implementation**:
```rust
// coordinator/src/vrf.rs
pub fn select_committee(&self, epoch: u64, verifier_pool: &[String]) -> Vec<String> {
    // VRF selects SUBSET of pool (e.g., 3 out of 10)
    pool.into_iter().take(self.committee_size).collect()
}

// coordinator/src/server.rs
// Parallel vote collection with timeout
for verifier_id in committee {
    tokio::spawn(async move {
        // Each verifier validates in parallel
        verifier_client.verify(request).await
    });
}
```

**Performance Profile**:
- **Committee Size**: 3 verifiers (not all 10!)
- **Execution**: Parallel async (Tokio)
- **Latency**: max(V1, V2, V3) + consensus_overhead
  - Verifier validation: ~50-100ms (TPM + IMA)
  - Consensus overhead: ~5ms
  - **Total**: ~55-105ms ✅

### Why Not All Verifiers?

**VRF Committee Selection** (vrf.rs:34-61):
1. **Pool**: 10 verifiers available
2. **Selected**: 3 verifiers per epoch (2f+1 for f=1)
3. **Unused**: 7 verifiers idle ✅

**Benefits**:
- ✅ Lower load per verifier
- ✅ Reduced network traffic
- ✅ Faster consensus (fewer votes)
- ✅ Cost savings (only 3 verifiers active)

---

## Optimization Strategies

### 1. Early Termination (조기 합의) 🚀 **HIGH IMPACT**

**Problem**: Current implementation waits for ALL votes or timeout (5s).

**Solution**: Terminate as soon as 2f+1 agreement is reached.

#### Implementation

```rust
// coordinator/src/consensus.rs - NEW METHOD
impl ConsensusManager {
    /// Check if early consensus is possible
    pub fn has_early_consensus(
        &self,
        votes: &HashMap<String, VerifierVote>,
    ) -> Option<VerificationResult> {
        let mut pass_count = 0;
        let mut fail_count = 0;

        for vote in votes.values() {
            match vote.result {
                VerificationResult::Pass => {
                    pass_count += 1;
                    if pass_count >= self.quorum {
                        return Some(VerificationResult::Pass); // Early exit!
                    }
                }
                VerificationResult::Fail => {
                    fail_count += 1;
                    if fail_count >= self.quorum {
                        return Some(VerificationResult::Fail); // Early exit!
                    }
                }
            }
        }
        None
    }
}
```

```rust
// coordinator/src/server.rs - MODIFIED VOTE COLLECTION
async fn collect_votes_with_early_termination(
    &self,
    committee: &[String],
    request: VerificationRequest,
) -> HashMap<String, VerifierVote> {
    let (tx, mut rx) = mpsc::channel(committee.len());
    let votes = Arc::new(Mutex::new(HashMap::new()));

    // Spawn verifier tasks
    for verifier_id in committee {
        let tx = tx.clone();
        let verifier_id = verifier_id.clone();

        tokio::spawn(async move {
            let vote = self.call_verifier(verifier_id, request).await;
            tx.send(vote).await.ok();
        });
    }
    drop(tx);

    // Collect votes with early termination
    let timeout = tokio::time::sleep(self.consensus_manager.vote_timeout());
    tokio::pin!(timeout);

    loop {
        tokio::select! {
            Some(vote) = rx.recv() => {
                let mut votes_guard = votes.lock().unwrap();
                votes_guard.insert(vote.verifier_id.clone(), vote);

                // Check early consensus
                if let Some(result) = self.consensus_manager.has_early_consensus(&votes_guard) {
                    info!("Early consensus reached: {:?}", result);
                    return votes_guard.clone(); // Exit early!
                }
            }
            _ = &mut timeout => {
                warn!("Vote collection timeout");
                break;
            }
            else => break,
        }
    }

    Arc::try_unwrap(votes).unwrap().into_inner().unwrap()
}
```

**Performance Gain**:
```
Before: Wait for all 3 votes or 5s timeout
  Best case: 100ms (slowest verifier)
  Worst case: 5000ms (timeout)

After: Exit when 2f+1 = 3 agree
  Best case: 50ms (fastest 3 verifiers agree) ← 50% improvement!
  Worst case: 100ms (need slowest verifier)
```

**Example**:
```
Committee: [V1, V2, V3]
Quorum: 3 (unanimous for f=1)

Time    Event
----    -----
0ms     → Request sent to V1, V2, V3
45ms    ← V1 responds: PASS
50ms    ← V2 responds: PASS
55ms    ← V3 responds: PASS  ← Exit here (3/3 consensus)
        ✅ Total: 55ms (don't wait for timeout!)
```

---

### 2. Speculative Execution 🔮 **MEDIUM IMPACT**

**Idea**: Pre-select next epoch's committee and warm up connections.

```rust
// coordinator/src/epoch_manager.rs - NEW
pub struct EpochManager {
    current_epoch: u64,
    current_committee: Vec<String>,
    next_committee: Vec<String>, // Pre-computed!
}

impl EpochManager {
    pub fn advance_epoch(&mut self, vrf: &VrfSelector, pool: &[String]) {
        // Current committee becomes active
        self.current_committee = self.next_committee.clone();

        // Pre-compute next committee
        let next_epoch = self.current_epoch + 1;
        self.next_committee = vrf.select_committee(next_epoch, pool);

        // Warm up connections to next committee
        self.warm_up_connections(&self.next_committee);
    }

    async fn warm_up_connections(&self, committee: &[String]) {
        for verifier_id in committee {
            // Establish gRPC connection in advance
            self.connection_pool.get_or_create(verifier_id).await;
        }
    }
}
```

**Performance Gain**:
```
Before: Connect to verifiers on-demand
  Latency: gRPC handshake (10-50ms) + validation

After: Pre-established connections
  Latency: validation only (save 10-50ms)
```

---

### 3. Adaptive Committee Size 📊 **LOW IMPACT**

**Idea**: Adjust committee size based on load.

```rust
// coordinator/src/adaptive.rs - NEW
pub struct AdaptiveCommitteeSelector {
    vrf: VrfSelector,
    load_monitor: LoadMonitor,
}

impl AdaptiveCommitteeSelector {
    pub fn select_optimal_committee(&self, pool: &[String]) -> Vec<String> {
        let load = self.load_monitor.current_load();

        let committee_size = if load < 0.3 {
            // Low load: Use larger committee for stronger security
            5 // f=2, more Byzantine tolerance
        } else if load < 0.7 {
            // Medium load: Standard
            3 // f=1
        } else {
            // High load: Minimal committee
            3 // f=1 minimum
        };

        self.vrf.select_committee_with_size(epoch, pool, committee_size)
    }
}
```

**Trade-off**:
- Low load: Stronger security (more verifiers)
- High load: Faster consensus (fewer verifiers)

---

### 4. Request Batching 📦 **HIGH IMPACT FOR THROUGHPUT**

**Idea**: Batch multiple metric submissions into single consensus round.

```rust
// coordinator/src/batcher.rs - NEW
pub struct RequestBatcher {
    batch_size: usize,
    batch_timeout: Duration,
    pending: Vec<MetricSubmission>,
}

impl RequestBatcher {
    pub async fn submit(&mut self, submission: MetricSubmission) -> SubmissionAck {
        self.pending.push(submission);

        // Batch when size reached or timeout
        if self.pending.len() >= self.batch_size
            || self.oldest_pending_age() > self.batch_timeout {
            self.flush_batch().await
        }
    }

    async fn flush_batch(&mut self) -> SubmissionAck {
        let batch = std::mem::take(&mut self.pending);

        // Single consensus round for entire batch
        let batch_merkle_root = compute_batch_merkle(batch);
        let consensus_result = self.run_consensus(batch_merkle_root).await;

        // Return individual acks
        self.generate_batch_acks(batch, consensus_result)
    }
}
```

**Performance Gain**:
```
Before: 100 requests × 100ms = 10,000ms total
After:  100 requests ÷ 10 per batch × 100ms = 1,000ms total
        ↑ 10x throughput improvement!
```

**Trade-off**:
- ✅ Much higher throughput
- ⚠️ Slightly higher latency per request (batching delay)

---

### 5. Vote Caching 💾 **MEDIUM IMPACT**

**Idea**: Cache verifier votes for identical requests.

```rust
// coordinator/src/vote_cache.rs - NEW
use lru::LruCache;

pub struct VoteCache {
    cache: LruCache<RequestHash, CachedConsensus>,
    ttl: Duration,
}

#[derive(Clone)]
struct CachedConsensus {
    result: VerificationResult,
    votes: HashMap<String, VerifierVote>,
    timestamp: Instant,
}

impl VoteCache {
    pub fn get(&self, request_hash: &RequestHash) -> Option<CachedConsensus> {
        self.cache.get(request_hash)
            .filter(|cached| cached.timestamp.elapsed() < self.ttl)
            .cloned()
    }

    pub fn insert(&mut self, request_hash: RequestHash, consensus: CachedConsensus) {
        self.cache.put(request_hash, consensus);
    }
}
```

**Performance Gain**:
```
Before: Every request goes through full consensus
After:  Duplicate requests served from cache (< 1ms)
```

**Use Case**: Idempotent retries, duplicate submissions

---

## Performance Comparison

### Latency Breakdown (3 Verifiers)

| Component | Baseline | Early Term | Speculative | Batch | Combined |
|-----------|----------|------------|-------------|-------|----------|
| gRPC Connect | 20ms | 20ms | **0ms** ✅ | 20ms | **0ms** |
| TPM Verify | 50ms | 50ms | 50ms | 50ms | 50ms |
| Vote Collect | 100ms | **50ms** ✅ | 100ms | 100ms | **50ms** |
| Consensus | 5ms | 5ms | 5ms | 5ms | 5ms |
| **Total** | **175ms** | **125ms** | **155ms** | **175ms** | **105ms** ✅ |

**Combined Optimization**: **40% latency reduction!**

---

## Throughput Comparison

| Strategy | Requests/sec | Speedup |
|----------|-------------|---------|
| Baseline | 10 | 1x |
| Early Termination | 14 | 1.4x |
| Batching (10x) | 100 | 10x ✅ |
| Combined | 140 | 14x ✅ |

---

## Recommended Configuration

### For Low Latency (Interactive)

```json
{
  "optimization": "early_termination",
  "committee_size": 3,
  "vote_timeout_secs": 2,
  "batch_size": 1,
  "speculative_execution": true
}
```

**Target**: < 100ms latency

### For High Throughput (Batch Processing)

```json
{
  "optimization": "batching",
  "committee_size": 3,
  "vote_timeout_secs": 5,
  "batch_size": 50,
  "batch_timeout_ms": 100,
  "vote_caching": true
}
```

**Target**: > 100 requests/sec

### For Maximum Security (Financial)

```json
{
  "optimization": "none",
  "committee_size": 7,
  "vote_timeout_secs": 10,
  "batch_size": 1,
  "require_unanimous": true
}
```

**Target**: Maximum Byzantine tolerance (f=3)

---

## Implementation Priority

### Phase 1: Quick Wins (1 week)
1. ✅ **Early Termination** - Modify `coordinator/src/server.rs`
2. ✅ **Speculative Execution** - Add `epoch_manager.rs`

**Impact**: 40% latency reduction

### Phase 2: Throughput (2 weeks)
1. ⏳ **Request Batching** - Add `batcher.rs`
2. ⏳ **Vote Caching** - Add `vote_cache.rs`

**Impact**: 10x throughput increase

### Phase 3: Advanced (1 month)
1. ⏳ **Adaptive Committee** - Dynamic sizing
2. ⏳ **Pipeline Processing** - Overlap rounds

**Impact**: Additional 2-3x improvement

---

## Monitoring

### Key Metrics

```rust
// Add to coordinator/src/metrics.rs
pub struct PerformanceMetrics {
    // Latency
    pub consensus_latency_p50: Duration,
    pub consensus_latency_p99: Duration,

    // Early termination effectiveness
    pub early_consensus_rate: f64, // % of requests that exit early
    pub avg_votes_needed: f64,     // Average votes to reach consensus

    // Throughput
    pub requests_per_second: f64,
    pub batch_size_avg: f64,

    // Cache
    pub cache_hit_rate: f64,
}
```

### Grafana Dashboard

```
Panel 1: Consensus Latency Distribution
  - P50, P95, P99 over time

Panel 2: Early Termination Rate
  - % of requests exiting early
  - Avg votes needed vs committee size

Panel 3: Throughput
  - Requests/sec
  - Batch sizes

Panel 4: Cache Efficiency
  - Hit rate
  - Cache size
```

---

## Conclusion

### Summary

| Optimization | Latency Impact | Throughput Impact | Complexity | Recommend |
|--------------|----------------|-------------------|------------|-----------|
| Early Termination | ⬇️ 30-50% | ⬆️ 30-50% | Low | ✅ **Yes** |
| Speculative Exec | ⬇️ 10-30% | → | Medium | ✅ **Yes** |
| Batching | → | ⬆️ 10x | Medium | ⚠️ Use case dependent |
| Vote Caching | ⬇️ 99% (cache hit) | ⬆️ 100x (cache hit) | Low | ✅ **Yes** |
| Adaptive Size | ⬇️ 0-20% | ⬆️ 0-50% | High | ⏳ Later |

### Final Recommendation

**Start with Early Termination + Speculative Execution**:
```
Expected Performance:
  - Latency: 55-80ms (vs 100ms baseline) ← 20-45% improvement
  - Throughput: 15-20 req/s (vs 10 req/s) ← 50-100% improvement
  - Implementation: 2-3 days
  - Risk: Low (non-breaking changes)
```

**Add Batching if throughput is critical**:
```
Expected Performance:
  - Throughput: 100-200 req/s ← 10-20x improvement
  - Latency: +10-50ms (batching delay) ← Trade-off
  - Use case: Background billing, not real-time
```

---

## References

- [gRPC Performance Best Practices](https://grpc.io/docs/guides/performance/)
- [Tokio Async Performance](https://tokio.rs/tokio/topics/performance)
- [BFT Consensus Optimizations](https://arxiv.org/abs/2003.02291) (HotStuff paper)
