#!/bin/bash
#
# generate-tpm-keys-simulator.sh
#
# Generate TPM 2.0 keys for BFT Agent using TPM simulator
#
# Prerequisites:
# - TPM simulator running (run setup-tpm-simulator.sh first)
# - Environment configured: source /etc/bft-tpm-env
#
# Usage:
#   source /etc/bft-tpm-env
#   ./generate-tpm-keys-simulator.sh [agent-id] [key-handle]
#
# Example:
#   ./generate-tpm-keys-simulator.sh agent-vllm-sim-001 0x81010001
#

set -e  # Exit on error

# Default values
AGENT_ID="${1:-agent-vllm-sim-001}"
KEY_HANDLE="${2:-0x81010001}"
OUTPUT_DIR="./tpm_keys/${AGENT_ID}"

echo "========================================"
echo "  TPM 2.0 Key Generation (Simulator)"
echo "========================================"
echo ""
echo "Agent ID:     ${AGENT_ID}"
echo "Key Handle:   ${KEY_HANDLE}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo ""

# Check if TPM simulator is accessible
echo "[1/7] Checking TPM simulator..."
if ! tpm2_getcap properties-fixed &> /dev/null; then
    echo "ERROR: Cannot communicate with TPM simulator"
    echo ""
    echo "Please ensure:"
    echo "  1. TPM simulator is running: systemctl status swtpm"
    echo "  2. Environment is set: source /etc/bft-tpm-env"
    echo "  3. Current TCTI: ${TPM2TOOLS_TCTI:-not set}"
    exit 1
fi
echo "✓ TPM simulator is accessible"
echo "  TCTI: ${TPM2TOOLS_TCTI}"

# Create output directory
echo ""
echo "[2/7] Creating output directory..."
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"
echo "✓ Output directory: $(pwd)"

# Create primary key in endorsement hierarchy
echo ""
echo "[3/7] Creating TPM primary key..."
tpm2_createprimary -C e -g sha256 -G rsa -c primary.ctx
if [ $? -eq 0 ]; then
    echo "✓ Primary key created: primary.ctx"
else
    echo "ERROR: Failed to create primary key"
    exit 1
fi

# Create signing key under primary
echo ""
echo "[4/7] Creating signing key..."
tpm2_create -C primary.ctx \
    -g sha256 \
    -G rsa:rsassa:null \
    -u tpm_key.pub \
    -r tpm_key.priv \
    -a "fixedtpm|fixedparent|sensitivedataorigin|userwithauth|sign"

if [ $? -eq 0 ]; then
    echo "✓ Signing key created:"
    echo "  - Public key: tpm_key.pub"
    echo "  - Private key: tpm_key.priv"
else
    echo "ERROR: Failed to create signing key"
    exit 1
fi

# Load key into TPM
echo ""
echo "[5/7] Loading key into TPM..."
tpm2_load -C primary.ctx \
    -u tpm_key.pub \
    -r tpm_key.priv \
    -c tpm_key.ctx

if [ $? -eq 0 ]; then
    echo "✓ Key loaded: tpm_key.ctx"
else
    echo "ERROR: Failed to load key"
    exit 1
fi

# Make key persistent
echo ""
echo "[6/7] Making key persistent at handle ${KEY_HANDLE}..."

# Check if handle already exists
if tpm2_readpublic -c ${KEY_HANDLE} &> /dev/null; then
    echo "⚠ Warning: Handle ${KEY_HANDLE} already exists"
    echo "  Releasing existing handle..."
    tpm2_evictcontrol -C o -c ${KEY_HANDLE} || true
fi

# Persist the key
tpm2_evictcontrol -C o -c tpm_key.ctx ${KEY_HANDLE}

if [ $? -eq 0 ]; then
    echo "✓ Key persisted at handle: ${KEY_HANDLE}"
else
    echo "ERROR: Failed to persist key"
    exit 1
fi

# Read public key for verification
echo ""
echo "[7/7] Exporting public key..."
tpm2_readpublic -c ${KEY_HANDLE} -o public_key.pem -f pem

if [ $? -eq 0 ]; then
    echo "✓ Public key exported: public_key.pem"
    echo ""
    echo "Public key (PEM format):"
    echo "----------------------------------------"
    cat public_key.pem
    echo "----------------------------------------"
else
    echo "⚠ Warning: Could not export public key to PEM"
fi

# Generate configuration file
echo ""
echo "Generating configuration..."
cat > agent_config.json <<EOF
{
  "agent_id": "${AGENT_ID}",
  "coordinator_endpoint": "http://localhost:50051",
  "signing_key_hex": "",
  "submission_interval_secs": 5,
  "mode": "file_monitor",
  "async_mode": true,
  "file_monitor_dir": "/tmp/vllm_metrics_queue",
  "file_monitor_poll_secs": 1,
  "file_monitor_batch_size": 10,
  "tpm": {
    "mode": "simulator",
    "device_path": "simulator:host=localhost,port=2321",
    "pcr_index": 16,
    "key_handle": "${KEY_HANDLE}"
  }
}
EOF

echo "✓ Configuration saved: agent_config.json"

# Test PCR operations
echo ""
echo "Testing PCR operations..."
echo "Current PCR 16 value:"
tpm2_pcrread sha256:16

echo ""
echo "Extending PCR 16 with test value..."
tpm2_pcrextend 16:sha256=0000000000000000000000000000000000000000000000000000000000000000

echo ""
echo "New PCR 16 value:"
tpm2_pcrread sha256:16

echo "✓ PCR operations working"

# Generate summary
echo ""
echo "========================================"
echo "  Key Generation Complete (Simulator)!"
echo "========================================"
echo ""
echo "Files created in: $(pwd)"
echo "  - primary.ctx          : Primary key context"
echo "  - tpm_key.pub          : Signing key public part"
echo "  - tpm_key.priv         : Signing key private part"
echo "  - tpm_key.ctx          : Loaded key context"
echo "  - public_key.pem       : Public key in PEM format"
echo "  - agent_config.json    : Agent configuration"
echo ""
echo "TPM Key Handle: ${KEY_HANDLE}"
echo "TPM Simulator:  ${TPM2TOOLS_TCTI}"
echo ""
echo "Next steps:"
echo "  1. Copy public_key.pem to verifiers"
echo "  2. Register public key in verifier config"
echo "  3. Build agent with TPM support:"
echo "     cargo build --release --features tpm-hardware"
echo "  4. Start agent:"
echo "     source /etc/bft-tpm-env"
echo "     AGENT_CONFIG=\$(pwd)/agent_config.json ./target/release/agent"
echo ""
echo "Verify TPM key:"
echo "  source /etc/bft-tpm-env"
echo "  tpm2_readpublic -c ${KEY_HANDLE}"
echo ""
echo "Monitor PCR 16:"
echo "  watch -n 1 'tpm2_pcrread sha256:16'"
echo ""
