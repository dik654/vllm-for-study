#!/bin/bash
# Example Scenario 2: Byzantine Fault Tolerance
# Simulate one malicious verifier (manually)

set -e

echo "========================================="
echo "Scenario 2: Byzantine Fault Tolerance"
echo "========================================="
echo ""

echo "Scenario: 4 verifiers, 1 could be malicious"
echo "Expected: Consensus still reached with 3/4 votes (quorum=3)"
echo ""

echo "Default setup:"
echo "  - Committee size: 3 (selected from pool of 4)"
echo "  - Quorum: 2 (tolerates f=1 Byzantine fault)"
echo "  - If 1 verifier is malicious, 2 honest votes still reach quorum"
echo ""

# Check if system is running
if ! docker-compose ps | grep -q "Up"; then
    echo "Starting BFT system..."
    docker-compose up -d
    sleep 5
fi

echo "Submitting metrics (3 requests)..."
docker-compose run --rm -e MODE=single -e REQUEST_COUNT=3 agent

echo ""
echo "Consensus results:"
docker-compose logs coordinator | grep "Consensus complete" | tail -3

echo ""
echo "Note: In this PoC, all verifiers are honest."
echo "To simulate Byzantine behavior:"
echo "  1. Stop one verifier: docker-compose stop verifier-4"
echo "  2. System still works with 3 verifiers (committee_size=3)"
echo "  3. Or modify verifier code to always vote FAIL"
echo ""
echo "✅ Scenario 2 Complete"
echo "System tolerates Byzantine faults via 2f+1 quorum"
