#!/bin/bash
#
# setup-tpm-simulator.sh
#
# Setup TPM 2.0 simulator for development/testing without real TPM hardware
#
# This script installs and configures:
# - swtpm: Software TPM 2.0 emulator
# - tpm2-abrmd: TPM 2.0 Access Broker & Resource Manager
# - tpm2-tools: Command-line tools for TPM 2.0
#
# Usage:
#   ./setup-tpm-simulator.sh
#

set -e  # Exit on error

echo "========================================"
echo "  TPM 2.0 Simulator Setup"
echo "========================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script requires root privileges"
    echo "Please run with sudo: sudo ./setup-tpm-simulator.sh"
    exit 1
fi

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    echo "Detected OS: $OS $VERSION_ID"
else
    echo "ERROR: Cannot detect OS"
    exit 1
fi

# Install packages based on OS
echo ""
echo "[1/5] Installing TPM simulator packages..."

if [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    echo "Installing on Ubuntu/Debian..."
    apt-get update
    apt-get install -y \
        swtpm \
        swtpm-tools \
        tpm2-abrmd \
        tpm2-tools \
        libtss2-dev

elif [ "$OS" = "fedora" ] || [ "$OS" = "rhel" ] || [ "$OS" = "centos" ]; then
    echo "Installing on Fedora/RHEL/CentOS..."
    dnf install -y \
        swtpm \
        swtpm-tools \
        tpm2-abrmd \
        tpm2-tools \
        tss2-devel

else
    echo "ERROR: Unsupported OS: $OS"
    echo "Please install manually:"
    echo "  - swtpm"
    echo "  - tpm2-abrmd"
    echo "  - tpm2-tools"
    echo "  - libtss2-dev"
    exit 1
fi

echo "✓ Packages installed"

# Create directory for TPM state
echo ""
echo "[2/5] Creating TPM state directory..."
TPM_STATE_DIR="/var/lib/bft-tpm-simulator"
mkdir -p "${TPM_STATE_DIR}"
chmod 700 "${TPM_STATE_DIR}"
echo "✓ TPM state directory: ${TPM_STATE_DIR}"

# Create systemd service for swtpm
echo ""
echo "[3/5] Creating swtpm systemd service..."
cat > /etc/systemd/system/swtpm.service <<EOF
[Unit]
Description=Software TPM 2.0 Emulator
After=network.target

[Service]
Type=simple
User=root
ExecStart=/usr/bin/swtpm socket \\
    --tpmstate dir=${TPM_STATE_DIR} \\
    --tpm2 \\
    --ctrl type=tcp,port=2322 \\
    --server type=tcp,port=2321 \\
    --flags not-need-init,startup-clear
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "✓ swtpm service created: /etc/systemd/system/swtpm.service"

# Create systemd service for tpm2-abrmd
echo ""
echo "[4/5] Creating tpm2-abrmd systemd service..."
cat > /etc/systemd/system/tpm2-abrmd-sim.service <<EOF
[Unit]
Description=TPM2 Access Broker and Resource Manager (Simulator)
After=swtpm.service
Requires=swtpm.service

[Service]
Type=dbus
BusName=com.intel.tss2.Tabrmd
ExecStart=/usr/sbin/tpm2-abrmd --tcti=swtpm:host=localhost,port=2321
User=root
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

echo "✓ tpm2-abrmd service created: /etc/systemd/system/tpm2-abrmd-sim.service"

# Reload systemd and start services
echo ""
echo "[5/5] Starting TPM simulator services..."
systemctl daemon-reload

echo "Starting swtpm..."
systemctl start swtpm
systemctl enable swtpm
sleep 2

echo "Starting tpm2-abrmd..."
systemctl start tpm2-abrmd-sim
systemctl enable tpm2-abrmd-sim
sleep 2

# Verify services
echo ""
echo "Verifying services..."
if systemctl is-active --quiet swtpm; then
    echo "✓ swtpm is running"
else
    echo "✗ swtpm failed to start"
    echo "  Check logs: journalctl -u swtpm -n 50"
    exit 1
fi

if systemctl is-active --quiet tpm2-abrmd-sim; then
    echo "✓ tpm2-abrmd is running"
else
    echo "✗ tpm2-abrmd failed to start"
    echo "  Check logs: journalctl -u tpm2-abrmd-sim -n 50"
    exit 1
fi

# Test TPM simulator
echo ""
echo "Testing TPM simulator..."
export TPM2TOOLS_TCTI="tabrmd:bus_type=session"

# Read PCR values
echo "Reading PCR values..."
tpm2_pcrread sha256:0,1,2,3

if [ $? -eq 0 ]; then
    echo "✓ TPM simulator is working!"
else
    echo "✗ TPM simulator test failed"
    exit 1
fi

# Create environment file
echo ""
echo "Creating environment configuration..."
cat > /etc/bft-tpm-env <<EOF
# BFT TPM Simulator Environment
# Source this file to use the TPM simulator with tpm2-tools

export TPM2TOOLS_TCTI="tabrmd:bus_type=session"
export TCTI=tabrmd:bus_type=session

# Alternative: Direct socket connection
# export TPM2TOOLS_TCTI="swtpm:host=localhost,port=2321"

echo "TPM simulator environment loaded"
echo "TCTI: \$TPM2TOOLS_TCTI"
EOF

echo "✓ Environment file created: /etc/bft-tpm-env"

# Generate summary
echo ""
echo "========================================"
echo "  TPM Simulator Setup Complete!"
echo "========================================"
echo ""
echo "Services started:"
echo "  - swtpm           : Software TPM 2.0 emulator"
echo "  - tpm2-abrmd-sim  : TPM Access Broker"
echo ""
echo "TPM state directory: ${TPM_STATE_DIR}"
echo ""
echo "To use the TPM simulator:"
echo "  1. Source environment file:"
echo "     source /etc/bft-tpm-env"
echo ""
echo "  2. Test TPM commands:"
echo "     tpm2_pcrread"
echo "     tpm2_getcap properties-fixed"
echo ""
echo "  3. Generate BFT agent keys:"
echo "     ./scripts/generate-tpm-keys-simulator.sh"
echo ""
echo "Service management:"
echo "  systemctl status swtpm"
echo "  systemctl status tpm2-abrmd-sim"
echo "  journalctl -u swtpm -f"
echo "  journalctl -u tpm2-abrmd-sim -f"
echo ""
echo "To stop services:"
echo "  systemctl stop tpm2-abrmd-sim swtpm"
echo ""
echo "To remove:"
echo "  systemctl disable --now tpm2-abrmd-sim swtpm"
echo "  rm /etc/systemd/system/swtpm.service"
echo "  rm /etc/systemd/system/tpm2-abrmd-sim.service"
echo "  rm -rf ${TPM_STATE_DIR}"
echo ""
