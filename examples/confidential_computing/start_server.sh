#!/bin/bash
# Start vLLM Confidential Inference Server
#
# Usage:
#   ./start_server.sh <encrypted_model_path> <key_id> [kms_provider]
#
# Examples:
#   # Local KMS (development)
#   ./start_server.sh ./llama-2-7b-encrypted my-secret-key
#
#   # AWS KMS (production)
#   ./start_server.sh ./llama-2-7b-encrypted \
#     arn:aws:kms:us-east-1:123456789:key/abc-123 aws_kms

set -e

ENCRYPTED_MODEL_PATH="${1}"
KEY_ID="${2}"
KMS_PROVIDER="${3:-local}"
HOST="${4:-0.0.0.0}"
PORT="${5:-8000}"

if [ -z "$ENCRYPTED_MODEL_PATH" ] || [ -z "$KEY_ID" ]; then
    echo "Usage: $0 <encrypted_model_path> <key_id> [kms_provider] [host] [port]"
    exit 1
fi

echo "=========================================="
echo "  vLLM Confidential Inference Server"
echo "=========================================="
echo ""
echo "Encrypted Model: $ENCRYPTED_MODEL_PATH"
echo "Key ID:          $KEY_ID"
echo "KMS Provider:    $KMS_PROVIDER"
echo "Host:            $HOST"
echo "Port:            $PORT"
echo ""

# Check TEE status
echo "Checking TEE status..."
python -m vllm.entrypoints.cli.confidential_cli status
echo ""

# Check if encrypted model exists
if [ ! -d "$ENCRYPTED_MODEL_PATH" ]; then
    echo "❌ ERROR: Encrypted model path not found: $ENCRYPTED_MODEL_PATH"
    exit 1
fi

# Start server
echo "Starting server..."
echo ""

python -m vllm.entrypoints.confidential_api_server \
    --confidential \
    --encrypted-model-path "$ENCRYPTED_MODEL_PATH" \
    --model-encryption-key-id "$KEY_ID" \
    --kms-provider "$KMS_PROVIDER" \
    --host "$HOST" \
    --port "$PORT"
