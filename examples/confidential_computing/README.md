# Confidential Computing Examples

This directory contains examples for using vLLM's confidential computing features.

## Examples

1. **simple_inference.py** - Basic confidential inference workflow
2. **encrypt_model.sh** - Script to encrypt model weights
3. **start_server.sh** - Start confidential inference server
4. **client_with_attestation.py** - Client that verifies server attestation

## Quick Start

### 1. Encrypt Your Model

```bash
bash encrypt_model.sh /path/to/your/model /path/to/encrypted/model my-key-id
```

### 2. Start Confidential Server

```bash
bash start_server.sh /path/to/encrypted/model my-key-id
```

### 3. Run Client

```bash
python client_with_attestation.py
```

## Requirements

- Python 3.8+
- TEE-enabled hardware:
  - **CPU TEE**: Intel TDX or AMD SEV-SNP
  - **GPU TEE**: NVIDIA Hopper (H100/H200) or Blackwell (B100/B200/RTX PRO 6000 Blackwell)
  - **Note**: Ada Lovelace GPUs (RTX 6000 Ada, L40/L40S) are NOT supported
- vLLM with confidential computing support
- KMS access (AWS KMS, Azure Key Vault, GCP KMS, or HashiCorp Vault)

## Setup

```bash
pip install vllm cryptography boto3 azure-identity azure-keyvault-keys google-cloud-kms hvac pynvml
```

## Notes

- These examples use local KMS for simplicity. In production, use a real KMS provider.
- Attestation verification is simplified in examples. Implement proper verification for production.
- Model paths and key IDs are placeholders. Replace with your actual values.
