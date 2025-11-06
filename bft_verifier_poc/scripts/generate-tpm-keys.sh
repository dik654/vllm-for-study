#!/bin/bash
#
# generate-tpm-keys.sh
#
# Generate TPM 2.0 keys for BFT Agent
#
# Prerequisites:
# - TPM 2.0 hardware (/dev/tpm0 or /dev/tpmrm0)
# - tpm2-tools installed (apt-get install tpm2-tools)
#
# Usage:
#   ./generate-tpm-keys.sh [agent-id] [key-handle]
#
# Example:
#   ./generate-tpm-keys.sh agent-vllm-001 0x81010001
#

set -e  # Exit on error

# Default values
AGENT_ID="${1:-agent-vllm-001}"
KEY_HANDLE="${2:-0x81010001}"
OUTPUT_DIR="./tpm_keys/${AGENT_ID}"

echo "========================================"
echo "  TPM 2.0 Key Generation for BFT Agent"
echo "========================================"
echo ""
echo "Agent ID:     ${AGENT_ID}"
echo "Key Handle:   ${KEY_HANDLE}"
echo "Output Dir:   ${OUTPUT_DIR}"
echo ""

# Check if TPM is available
echo "[1/7] Checking TPM availability..."
if [ ! -e "/dev/tpm0" ] && [ ! -e "/dev/tpmrm0" ]; then
    echo "ERROR: No TPM device found (/dev/tpm0 or /dev/tpmrm0)"
    echo "Please ensure TPM 2.0 is enabled in BIOS and kernel driver is loaded"
    exit 1
fi

if [ -e "/dev/tpmrm0" ]; then
    TPM_DEVICE="/dev/tpmrm0"
    echo "✓ TPM resource manager found: ${TPM_DEVICE}"
else
    TPM_DEVICE="/dev/tpm0"
    echo "✓ TPM device found: ${TPM_DEVICE}"
    echo "  Note: Using direct access. Resource manager (/dev/tpmrm0) is recommended."
fi

# Check if tpm2-tools is installed
echo ""
echo "[2/7] Checking tpm2-tools installation..."
if ! command -v tpm2_createprimary &> /dev/null; then
    echo "ERROR: tpm2-tools not found"
    echo "Please install: sudo apt-get install tpm2-tools"
    exit 1
fi
echo "✓ tpm2-tools found: $(tpm2_createprimary --version | head -1)"

# Create output directory
echo ""
echo "[3/7] Creating output directory..."
mkdir -p "${OUTPUT_DIR}"
cd "${OUTPUT_DIR}"
echo "✓ Output directory: $(pwd)"

# Create primary key in endorsement hierarchy
echo ""
echo "[4/7] Creating TPM primary key..."
tpm2_createprimary -C e -g sha256 -G rsa -c primary.ctx
if [ $? -eq 0 ]; then
    echo "✓ Primary key created: primary.ctx"
else
    echo "ERROR: Failed to create primary key"
    exit 1
fi

# Create signing key under primary
echo ""
echo "[5/7] Creating signing key..."
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
echo "[6/7] Loading key into TPM..."
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
echo "[7/7] Making key persistent at handle ${KEY_HANDLE}..."

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
echo "Reading public key..."
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
  "coordinator_endpoint": "http://coordinator.example.com:50051",
  "signing_key_hex": "",
  "submission_interval_secs": 5,
  "mode": "file_monitor",
  "async_mode": true,
  "file_monitor_dir": "/tmp/vllm_metrics_queue",
  "file_monitor_poll_secs": 1,
  "file_monitor_batch_size": 10,
  "tpm": {
    "mode": "hardware",
    "device_path": "${TPM_DEVICE}",
    "pcr_index": 16,
    "key_handle": "${KEY_HANDLE}"
  }
}
EOF

echo "✓ Configuration saved: agent_config.json"

# Generate summary
echo ""
echo "========================================"
echo "  TPM Key Generation Complete!"
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
echo ""
echo "Next steps:"
echo "  1. Copy public_key.pem to verifiers"
echo "  2. Register public key in verifier config"
echo "  3. Use agent_config.json to start agent:"
echo "     AGENT_CONFIG=\$(pwd)/agent_config.json ./target/release/agent"
echo ""
echo "Verify TPM key:"
echo "  tpm2_readpublic -c ${KEY_HANDLE}"
echo ""
echo "Test PCR extend:"
echo "  tpm2_pcrread sha256:16"
echo "  tpm2_pcrextend 16:sha256=0000000000000000000000000000000000000000000000000000000000000000"
echo "  tpm2_pcrread sha256:16"
echo ""
