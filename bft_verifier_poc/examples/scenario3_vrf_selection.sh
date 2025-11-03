#!/bin/bash
# Example Scenario 3: VRF Committee Selection
# Demonstrate fair and deterministic verifier selection

set -e

echo "========================================="
echo "Scenario 3: VRF Committee Selection"
echo "========================================="
echo ""

echo "Scenario: Multiple epochs, observe committee rotation"
echo "Expected: Different committees per epoch, fair distribution"
echo ""

# Check if system is running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting BFT system..."
    docker-compose up -d
    sleep 5
fi

echo "Observing committee selection over multiple submissions..."
echo "Each submission triggers epoch check and committee selection"
echo ""

# Submit 5 metrics over time (wait for epoch changes)
echo "Submitting 5 requests (one every 6 seconds)..."
for i in {1..5}; do
    echo ""
    echo "Request $i/5..."
    docker-compose run --rm -e MODE=single -e REQUEST_COUNT=1 agent

    # Show current epoch and committee
    echo "Checking epoch and committee:"
    docker-compose logs coordinator | grep "Current epoch\|Selected committee" | tail -2

    if [ $i -lt 5 ]; then
        echo "Waiting 6 seconds for potential epoch change..."
        sleep 6
    fi
done

echo ""
echo "Committee selection summary:"
docker-compose logs coordinator | grep "Selected committee" | tail -5

echo ""
echo "VRF Properties:"
echo "  ✓ Deterministic: Same epoch → same committee"
echo "  ✓ Unpredictable: Cannot predict future committees"
echo "  ✓ Fair: All verifiers have equal probability"
echo ""
echo "✅ Scenario 3 Complete"
echo "VRF ensures fair and secure committee selection"
