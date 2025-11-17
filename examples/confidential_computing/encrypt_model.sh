#!/bin/bash
# Encrypt model weights for confidential computing
#
# Usage:
#   ./encrypt_model.sh <model_path> <output_path> <key_id> [kms_provider]
#
# Examples:
#   # Local encryption (development)
#   ./encrypt_model.sh ./llama-2-7b ./llama-2-7b-encrypted my-secret-key
#
#   # AWS KMS (production)
#   ./encrypt_model.sh ./llama-2-7b ./llama-2-7b-encrypted \
#     arn:aws:kms:us-east-1:123456789:key/abc-123 aws_kms

set -e

MODEL_PATH="${1}"
OUTPUT_PATH="${2}"
KEY_ID="${3}"
KMS_PROVIDER="${4:-local}"

if [ -z "$MODEL_PATH" ] || [ -z "$OUTPUT_PATH" ] || [ -z "$KEY_ID" ]; then
    echo "Usage: $0 <model_path> <output_path> <key_id> [kms_provider]"
    exit 1
fi

echo "=========================================="
echo "  vLLM Model Encryption"
echo "=========================================="
echo ""
echo "Model Path:    $MODEL_PATH"
echo "Output Path:   $OUTPUT_PATH"
echo "Key ID:        $KEY_ID"
echo "KMS Provider:  $KMS_PROVIDER"
echo ""

if [ "$KMS_PROVIDER" = "local" ]; then
    echo "⚠️  WARNING: Using local KMS (NOT secure for production)"
    echo ""
fi

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ ERROR: Model path not found: $MODEL_PATH"
    exit 1
fi

# Run encryption
python -m vllm.entrypoints.cli.confidential_cli encrypt \
    "$MODEL_PATH" \
    "$OUTPUT_PATH" \
    --key-id "$KEY_ID" \
    --kms-provider "$KMS_PROVIDER"

echo ""
echo "=========================================="
echo "  Encryption Complete!"
echo "=========================================="
echo ""
echo "Your encrypted model is ready at:"
echo "  $OUTPUT_PATH"
echo ""
echo "To use this model with vLLM:"
echo "  python -m vllm.entrypoints.confidential_api_server \\"
echo "    --confidential \\"
echo "    --encrypted-model-path $OUTPUT_PATH \\"
echo "    --model-encryption-key-id $KEY_ID \\"
echo "    --kms-provider $KMS_PROVIDER"
echo ""
