#!/bin/bash
# Example Scenario 1: Honest Consensus
# All verifiers vote PASS on valid metrics

set -e

echo "========================================="
echo "Scenario 1: Honest Consensus (All PASS)"
echo "========================================="
echo ""

# Check if system is running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting BFT system..."
    docker-compose up -d
    sleep 5
fi

echo "Scenario: Agent submits valid metrics"
echo "Expected: All 3 verifiers vote PASS, consensus reached"
echo ""

# Submit a single valid metric
docker-compose run --rm -e MODE=single -e REQUEST_COUNT=1 agent

echo ""
echo "Check coordinator logs for consensus result:"
docker-compose logs coordinator | grep "Consensus complete" | tail -1

echo ""
echo "Check verifier votes:"
for i in {1..3}; do
    echo "Verifier-$i:"
    docker-compose logs verifier-$i | grep "Verification PASSED" | tail -1
done

echo ""
echo "✅ Scenario 1 Complete"
echo "All verifiers should have voted PASS"
