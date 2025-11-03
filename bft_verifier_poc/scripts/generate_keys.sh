#!/bin/bash
# Key generation helper for BFT Verifier PoC
# Generates agent and verifier keypairs

set -e

echo "========================================="
echo "  BFT Verifier - Key Generation"
echo "========================================="
echo ""

# Check if Rust is available
if ! command -v cargo &> /dev/null; then
    echo "Error: Rust is not installed"
    echo "This script requires Rust to build the agent"
    exit 1
fi

# Build agent if not already built
if [ ! -f "target/release/agent" ]; then
    echo "Building agent binary..."
    cargo build --release --package bft-agent --quiet
    echo "✓ Build complete"
    echo ""
fi

# Generate keys for specified number of agents
NUM_AGENTS=${1:-1}

echo "Generating keys for $NUM_AGENTS agent(s)..."
echo ""

for i in $(seq 1 $NUM_AGENTS); do
    AGENT_ID="agent-$i"
    echo "=== $AGENT_ID ==="

    # Run agent once to generate keys
    OUTPUT=$(RUST_LOG=error AGENT_ID=$AGENT_ID MODE=single REQUEST_COUNT=0 \
             COORDINATOR_ENDPOINT=http://localhost:1 \
             ./target/release/agent 2>&1 || true)

    # Extract public key from output
    PUBLIC_KEY=$(echo "$OUTPUT" | grep "Public key (hex):" | awk '{print $NF}')

    if [ -z "$PUBLIC_KEY" ]; then
        echo "Error: Failed to generate keys"
        continue
    fi

    echo "Agent ID: $AGENT_ID"
    echo "Public Key: $PUBLIC_KEY"
    echo ""
    echo "Add to verifier config:"
    echo "  \"agent_public_keys\": {"
    echo "    \"$AGENT_ID\": \"$PUBLIC_KEY\""
    echo "  }"
    echo ""
done

echo "========================================="
echo "Key Generation Complete"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. Copy public keys from above"
echo "  2. Add to config/verifier_config.json"
echo "  3. Restart verifiers with updated config"
echo ""
echo "Or use environment variable:"
echo "  AGENT_KEYS_JSON='{\"agent-1\":\"$PUBLIC_KEY\"}'"
