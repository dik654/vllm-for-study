"""
vLLM Confidential Computing Module

Enables secure, private LLM inference using Trusted Execution Environments (TEE).
Supports Intel TDX, AMD SEV-SNP, and NVIDIA H100 Confidential GPU.

Key Features:
- Remote attestation for verifiable security
- Encrypted model weight upload and storage
- Secure key management with KMS integration
- CPU+GPU unified TEE support
- End-to-end encryption for prompts and outputs
"""

from vllm.confidential.attestation import (
    AttestationManager,
    AttestationReport,
    verify_attestation,
)
from vllm.confidential.encryption import (
    EncryptionManager,
    ModelEncryption,
    decrypt_model_weights,
    encrypt_model_weights,
)
from vllm.confidential.kms import (
    KMSClient,
    KMSProvider,
)
from vllm.confidential.tee_manager import (
    TEEManager,
    TEEPlatform,
    get_tee_platform,
)

__all__ = [
    "AttestationManager",
    "AttestationReport",
    "verify_attestation",
    "EncryptionManager",
    "ModelEncryption",
    "decrypt_model_weights",
    "encrypt_model_weights",
    "KMSClient",
    "KMSProvider",
    "TEEManager",
    "TEEPlatform",
    "get_tee_platform",
]
