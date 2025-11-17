# vLLM Confidential Computing

**Secure, Private LLM Inference with Trusted Execution Environments (TEE)**

## Overview

vLLM Confidential Computing enables users to run private AI inference workloads with hardware-enforced security guarantees. Your model weights, prompts, and outputs are protected even from cloud providers and system administrators.

### Key Features

- 🔒 **Hardware-Enforced Security**: Uses Intel TDX, AMD SEV-SNP, and NVIDIA Confidential GPU (Hopper/Blackwell)
- 🔐 **Encrypted Model Weights**: Envelope encryption with KMS integration
- ✅ **Remote Attestation**: Cryptographic proof of TEE execution
- 🚀 **Low Overhead**: 4-8% performance penalty on GPU TEE
- 🔑 **Multi-KMS Support**: AWS KMS, Azure Key Vault, GCP KMS, HashiCorp Vault

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User's Environment                     │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │ Private Model│────────▶│  Encrypt     │                 │
│  │   Weights    │         │  with KMS    │                 │
│  └──────────────┘         └──────┬───────┘                 │
│                                   │                          │
│                                   │ Encrypted Model          │
└───────────────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────┐
│              TEE Environment (Intel TDX / AMD SEV)          │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         CPU TEE (Intel TDX / AMD SEV-SNP)            │  │
│  │                                                       │  │
│  │  ┌────────────┐      ┌──────────────┐               │  │
│  │  │ Decrypt    │─────▶│ vLLM Engine  │               │  │
│  │  │ with KMS   │      │              │               │  │
│  │  └────────────┘      └──────┬───────┘               │  │
│  │                             │                        │  │
│  │                             │ Secure Memory Transfer │  │
│  └─────────────────────────────┼────────────────────────┘  │
│                                │                            │
│  ┌─────────────────────────────┼────────────────────────┐  │
│  │   GPU TEE (NVIDIA Hopper/Blackwell Confidential)   │  │
│  │                             │                        │  │
│  │                             ▼                        │  │
│  │                   ┌──────────────────┐              │  │
│  │                   │   Model Inference │              │  │
│  │                   │   (Protected)     │              │  │
│  │                   └──────────────────┘              │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │           Remote Attestation                          │  │
│  │  ┌──────────┐  ┌────────┐  ┌────────────┐            │  │
│  │  │ Generate │─▶│ Verify │─▶│ KMS Release│            │  │
│  │  │ Report   │  │        │  │    Keys    │            │  │
│  │  └──────────┘  └────────┘  └────────────┘            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Check TEE Status

```bash
vllm-confidential status
# Or: python -m vllm.entrypoints.cli.confidential_cli status
```

Output:
```
╔══════════════════════════════════════════════════════╗
║     vLLM Confidential Computing Status              ║
╚══════════════════════════════════════════════════════╝

Platform:             intel_tdx
Confidential:         ✓ YES

Features:
  ✓ Memory Encryption
  ✓ Remote Attestation
  ✓ Gpu Tee
  ✓ Secure Boot
```

### 2. Encrypt Your Model

```bash
# Using local KMS (development only)
vllm-confidential encrypt \
  /path/to/your/model \
  /path/to/encrypted/model \
  --key-id my-secret-key

# Using AWS KMS (production)
vllm-confidential encrypt \
  /path/to/your/model \
  /path/to/encrypted/model \
  --key-id arn:aws:kms:us-east-1:123456789:key/abc-123 \
  --kms-provider aws_kms \
  --kms-config '{"region": "us-east-1"}'
```

### 3. Start Confidential Inference Server

```bash
python -m vllm.entrypoints.confidential_api_server \
  --confidential \
  --encrypted-model-path /path/to/encrypted/model \
  --model-encryption-key-id my-secret-key \
  --kms-provider aws_kms \
  --host 0.0.0.0 \
  --port 8000
```

### 4. Get Attestation Report

```bash
curl http://localhost:8000/v1/confidential/attestation \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"nonce": "random_challenge_123"}'
```

### 5. Run Inference

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ]
  }'
```

## Supported TEE Platforms

### CPU TEE Platforms

| Platform | Architecture | Status |
|----------|--------------|--------|
| Intel TDX | Trust Domain Extensions | ✅ Supported |
| AMD SEV-SNP | Secure Encrypted Virtualization | ✅ Supported |

### GPU TEE Platforms

| GPU Model | Architecture | Confidential Computing | Status |
|-----------|--------------|------------------------|--------|
| **Hopper GPUs** | | | |
| H100 Tensor Core | Hopper | ✅ Yes | Supported |
| H200 | Hopper | ✅ Yes | Supported |
| **Blackwell GPUs** | | | |
| B100 | Blackwell | ✅ Yes | Supported (2025) |
| B200 | Blackwell | ✅ Yes | Supported (2025) |
| RTX PRO 6000 Blackwell Server Edition | Blackwell | ✅ Yes | Supported (2025) |
| **Ada Lovelace GPUs** | | | |
| RTX 6000 Ada | Ada Lovelace | ❌ No | Not Supported |
| L40 / L40S | Ada Lovelace | ❌ No | Not Supported |

**Note**: Only **Hopper** and **Blackwell** architecture GPUs support Confidential Computing. Ada Lovelace GPUs (RTX 6000 Ada, L40, L40S) do **NOT** support Confidential Computing.

## KMS Providers

### AWS KMS

```bash
export AWS_REGION=us-east-1
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key

vllm-confidential encrypt \
  ./model ./encrypted_model \
  --key-id arn:aws:kms:us-east-1:123456789:key/abc-123 \
  --kms-provider aws_kms
```

### Azure Key Vault

```bash
export AZURE_KEY_VAULT_URL=https://your-vault.vault.azure.net

vllm-confidential encrypt \
  ./model ./encrypted_model \
  --key-id your-key-name \
  --kms-provider azure_key_vault
```

### GCP KMS

```bash
export GCP_PROJECT_ID=your-project-id
export GCP_LOCATION=global

vllm-confidential encrypt \
  ./model ./encrypted_model \
  --key-id keyRingId/cryptoKeyId \
  --kms-provider gcp_kms
```

### HashiCorp Vault

```bash
export VAULT_ADDR=https://vault.example.com
export VAULT_TOKEN=your-token

vllm-confidential encrypt \
  ./model ./encrypted_model \
  --key-id transit-key-name \
  --kms-provider hashicorp_vault
```

## API Endpoints

### Confidential Computing Status

```bash
GET /v1/confidential/status
```

Returns TEE platform information and available features.

### Generate Attestation

```bash
POST /v1/confidential/attestation
{
  "nonce": "optional_challenge"
}
```

Returns cryptographic attestation report proving TEE execution.

### Verify Attestation

```bash
POST /v1/confidential/verify-attestation
{
  "attestation": { ... },
  "expected_measurement": "optional_hash"
}
```

Verifies an attestation report.

### Decrypt Model Inside TEE

```bash
POST /v1/confidential/models/decrypt
{
  "encrypted_path": "/path/to/encrypted/model",
  "output_path": "/tmp/decrypted",
  "key_id": "kms_key_id"
}
```

Decrypts model weights inside the TEE (requires attestation).

## Security Guarantees

### What is Protected

✅ **Model Weights**: Encrypted at rest and in transit, decrypted only inside TEE
✅ **User Prompts**: Never exposed outside TEE
✅ **Model Outputs**: Generated inside TEE, encrypted before leaving
✅ **Inference Process**: All computation happens in hardware-protected memory

### What is NOT Protected

❌ **Model Architecture**: Public (needed for inference)
❌ **API Endpoints**: Metadata visible (use TLS for transport security)
❌ **Timing Information**: Side-channel attacks may infer some information

### Trust Model

You must trust:
- Hardware manufacturer (Intel, AMD, NVIDIA)
- TEE firmware and microcode
- KMS provider (for key management)

You do NOT need to trust:
- Cloud provider / datacenter operator
- System administrator
- Other VMs on the same host

## Performance

Based on recent research and Phala Network benchmarks:

| Workload | CPU TEE Overhead | GPU TEE Overhead |
|----------|------------------|------------------|
| Small models (<7B) | <10% | 4-8% |
| Medium models (7-32B) | <10% | 4-8% |
| Large models (>32B) | <5% | <5% |
| Long context (>8k tokens) | <1% | <1% |

## CLI Reference

```bash
# Show status
vllm-confidential status

# Generate attestation
vllm-confidential attestation --nonce challenge123 -o report.json

# Verify attestation
vllm-confidential verify report.json

# Encrypt model
vllm-confidential encrypt \
  <model_path> <output_path> \
  --key-id <kms_key_id> \
  --kms-provider <provider>

# Decrypt model (inside TEE only)
vllm-confidential decrypt \
  <encrypted_path> <output_path> \
  --key-id <kms_key_id> \
  --kms-provider <provider>
```

## Examples

See `examples/confidential_computing/` for complete examples:

- `simple_inference.py` - Basic confidential inference
- `mutual_attestation.py` - Client and server mutual attestation
- `encrypted_model_upload.py` - Uploading encrypted models
- `kms_integration.py` - KMS provider examples

## Troubleshooting

### "Not running in TEE" Warning

**Problem**: System reports not running in TEE
**Solution**:
- Ensure you're running on compatible hardware (Intel TDX, AMD SEV-SNP)
- Check BIOS settings for TEE enablement
- Verify kernel and firmware support

### KMS Connection Failed

**Problem**: Cannot connect to KMS
**Solution**:
- Check network connectivity to KMS endpoint
- Verify credentials and permissions
- Ensure KMS key exists and has correct policy

### Attestation Generation Failed

**Problem**: Cannot generate attestation report
**Solution**:
- Check TEE device files (`/dev/tdx_guest` or `/dev/sev-guest`)
- Verify TEE kernel modules are loaded
- Check system logs for TEE errors

## References

- [Intel TDX Documentation](https://www.intel.com/content/www/us/en/developer/tools/trust-domain-extensions/overview.html)
- [AMD SEV-SNP White Paper](https://www.amd.com/en/developer/sev.html)
- [NVIDIA Confidential Computing](https://www.nvidia.com/en-us/data-center/solutions/confidential-computing/)
- [Phala Network GPU TEE](https://docs.phala.network/confidential-ai-inference/host-llm-in-tee)
- [BlindLlama](https://blindllama.mithrilsecurity.io/)

## Contributing

We welcome contributions! Areas for improvement:

- [ ] Additional KMS provider support
- [ ] Client-side encryption SDK
- [ ] Attestation policy framework
- [ ] Performance optimizations
- [ ] Additional TEE platform support

## License

Apache 2.0 - Same as vLLM

## Support

For issues and questions:
- GitHub Issues: https://github.com/vllm-project/vllm/issues
- Discord: https://discord.gg/vllm

---

**⚠️ Security Note**: This is an initial implementation. For production use, ensure:
1. Proper attestation verification
2. Secure KMS configuration
3. Regular security audits
4. Up-to-date TEE firmware
