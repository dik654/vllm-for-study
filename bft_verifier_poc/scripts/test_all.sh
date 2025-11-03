#!/bin/bash
# Comprehensive test script for BFT Verifier PoC

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  BFT Verifier PoC - Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

FAILED=0
PASSED=0

# Function to run a test
run_test() {
    local name=$1
    local command=$2

    echo -e "${BLUE}Testing: $name${NC}"
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        PASSED=$((PASSED + 1))
    else
        echo -e "${RED}✗ FAILED${NC}"
        FAILED=$((FAILED + 1))
    fi
    echo ""
}

# Check Rust installation
echo "=== Environment Checks ==="
run_test "Rust installed" "command -v cargo"
run_test "Docker installed" "command -v docker"
run_test "Docker Compose installed" "command -v docker-compose"

# Run unit tests
echo "=== Unit Tests ==="
if command -v cargo &> /dev/null; then
    echo -e "${BLUE}Running Rust unit tests...${NC}"
    if cargo test --workspace --quiet 2>&1 | tee test_output.log; then
        TOTAL_TESTS=$(grep -o "test result: ok" test_output.log | wc -l)
        echo -e "${GREEN}✓ All unit tests passed${NC}"
        echo "  Total tests: $TOTAL_TESTS"
        PASSED=$((PASSED + 1))
        rm test_output.log
    else
        echo -e "${RED}✗ Some unit tests failed${NC}"
        FAILED=$((FAILED + 1))
    fi
else
    echo -e "${YELLOW}⊘ Skipping unit tests (Rust not available)${NC}"
fi
echo ""

# Check code formatting
echo "=== Code Quality ==="
if command -v cargo &> /dev/null; then
    run_test "Cargo check" "cargo check --workspace --quiet"
else
    echo -e "${YELLOW}⊘ Skipping cargo check (Rust not available)${NC}"
fi

# Check documentation
echo "=== Documentation Checks ==="
run_test "README exists" "test -f README.md"
run_test "SPECIFICATION exists" "test -f SPECIFICATION.md"
run_test "TODO exists" "test -f TODO.md"
run_test "VALIDATION exists" "test -f VALIDATION.md"
run_test "SUMMARY exists" "test -f SUMMARY.md"
run_test "Docker Compose config" "test -f docker-compose.yml"
run_test "Dockerfile exists" "test -f Dockerfile"

# Check configuration examples
echo "=== Configuration Checks ==="
run_test "Coordinator config example" "test -f config/examples/coordinator_config.json"
run_test "Verifier config example" "test -f config/examples/verifier_config.json"
run_test "Agent config example" "test -f config/examples/agent_config.json"

# Check example scripts
echo "=== Example Scripts ==="
run_test "Scenario 1 script" "test -x examples/scenario1_honest_consensus.sh"
run_test "Scenario 2 script" "test -x examples/scenario2_byzantine_tolerance.sh"
run_test "Scenario 3 script" "test -x examples/scenario3_vrf_selection.sh"
run_test "Scenario 4 script" "test -x examples/scenario4_performance.sh"
run_test "Quickstart script" "test -x quickstart.sh"

# Summary
echo "========================================="
echo -e "${BLUE}Test Summary${NC}"
echo "========================================="
echo -e "${GREEN}Passed: $PASSED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}Failed: $FAILED${NC}"
    exit 1
else
    echo -e "${GREEN}Failed: 0${NC}"
    echo ""
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
fi
