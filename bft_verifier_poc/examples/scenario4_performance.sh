#!/bin/bash
# Example Scenario 4: Performance Benchmark
# Measure consensus latency and throughput

set -e

echo "========================================="
echo "Scenario 4: Performance Benchmark"
echo "========================================="
echo ""

# Check if system is running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting BFT system..."
    docker-compose up -d
    sleep 5
fi

echo "Benchmark Setup:"
echo "  - 3 verifiers selected per epoch"
echo "  - Parallel vote collection"
echo "  - Target: <100ms consensus latency"
echo ""

# Benchmark: 20 sequential requests
echo "Running benchmark: 20 sequential requests..."
echo "Start time: $(date +%s)"

START=$(date +%s)
docker-compose run --rm -e MODE=single -e REQUEST_COUNT=20 -e SUBMISSION_INTERVAL_SECS=0 agent
END=$(date +%s)

DURATION=$((END - START))
echo ""
echo "End time: $END"
echo "Total duration: ${DURATION}s for 20 requests"
echo "Average: $((DURATION * 1000 / 20))ms per request"

echo ""
echo "Consensus latencies (from logs):"
docker-compose logs coordinator | grep "Consensus complete" | tail -20 | \
    awk '{print $1, $2}' | \
    while read line; do echo "  $line"; done

echo ""
echo "Performance Summary:"
echo "  - Total requests: 20"
echo "  - Duration: ${DURATION}s"
echo "  - Throughput: $((20 * 100 / DURATION))% of 1 req/sec"
echo ""
echo "Note: Actual performance depends on:"
echo "  - Network latency (Docker bridge)"
echo "  - System resources"
echo "  - Vote timeout configuration"
echo ""
echo "✅ Scenario 4 Complete"
echo "See coordinator logs for detailed timing"
