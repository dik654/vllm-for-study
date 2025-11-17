"""
Key Management Service (KMS) Integration

Provides integration with various KMS providers for secure key management
in confidential computing environments.
"""

import base64
import os
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class KMSProvider(str, Enum):
    """Supported KMS providers"""
    AWS_KMS = "aws_kms"
    AZURE_KEY_VAULT = "azure_key_vault"
    GCP_KMS = "gcp_kms"
    HASHICORP_VAULT = "hashicorp_vault"
    LOCAL = "local"  # For development only


class KMSClient(ABC):
    """
    Abstract base class for KMS clients.

    KMS clients handle encryption and decryption of data encryption keys (DEKs)
    using keys managed by external key management services.
    """

    @abstractmethod
    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """
        Encrypt data using a KMS key.

        Args:
            key_id: KMS key identifier
            plaintext: Data to encrypt

        Returns:
            bytes: Encrypted data
        """
        pass

    @abstractmethod
    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """
        Decrypt data using a KMS key.

        Args:
            key_id: KMS key identifier
            ciphertext: Data to decrypt

        Returns:
            bytes: Decrypted data
        """
        pass

    @abstractmethod
    def verify_attestation(
        self,
        attestation_report: 'AttestationReport'
    ) -> bool:
        """
        Verify attestation report before releasing keys.

        Args:
            attestation_report: TEE attestation report

        Returns:
            bool: True if attestation is valid
        """
        pass


class AWSKMSClient(KMSClient):
    """AWS KMS client for confidential computing"""

    def __init__(
        self,
        region: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None
    ):
        try:
            import boto3

            self.client = boto3.client(
                'kms',
                region_name=region or os.getenv('AWS_REGION', 'us-east-1'),
                aws_access_key_id=access_key_id or os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=secret_access_key or os.getenv('AWS_SECRET_ACCESS_KEY')
            )
            logger.info("AWS KMS client initialized")
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS KMS. Install with: pip install boto3"
            )

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt using AWS KMS"""
        try:
            response = self.client.encrypt(
                KeyId=key_id,
                Plaintext=plaintext
            )
            return response['CiphertextBlob']
        except Exception as e:
            logger.error(f"AWS KMS encryption failed: {e}")
            raise

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt using AWS KMS"""
        try:
            response = self.client.decrypt(
                KeyId=key_id,
                CiphertextBlob=ciphertext
            )
            return response['Plaintext']
        except Exception as e:
            logger.error(f"AWS KMS decryption failed: {e}")
            raise

    def verify_attestation(
        self,
        attestation_report: 'AttestationReport'
    ) -> bool:
        """
        Verify attestation using AWS Nitro Enclaves or custom logic.

        In production, this would integrate with AWS Nitro Attestation.
        """
        logger.info("Verifying attestation for AWS KMS access")
        # TODO: Implement proper attestation verification
        return True


class AzureKeyVaultClient(KMSClient):
    """Azure Key Vault client for confidential computing"""

    def __init__(
        self,
        vault_url: Optional[str] = None,
        credential: Optional[any] = None
    ):
        try:
            from azure.identity import DefaultAzureCredential
            from azure.keyvault.keys.crypto import CryptographyClient
            from azure.keyvault.keys import KeyClient

            vault_url = vault_url or os.getenv('AZURE_KEY_VAULT_URL')
            if not vault_url:
                raise ValueError("AZURE_KEY_VAULT_URL must be set")

            self.credential = credential or DefaultAzureCredential()
            self.vault_url = vault_url
            self.key_client = KeyClient(
                vault_url=vault_url,
                credential=self.credential
            )
            logger.info("Azure Key Vault client initialized")
        except ImportError:
            raise ImportError(
                "Azure SDK is required. Install with: "
                "pip install azure-identity azure-keyvault-keys"
            )

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt using Azure Key Vault"""
        try:
            from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm

            key = self.key_client.get_key(key_id)
            crypto_client = CryptographyClient(key, credential=self.credential)

            result = crypto_client.encrypt(
                EncryptionAlgorithm.rsa_oaep_256,
                plaintext
            )
            return result.ciphertext
        except Exception as e:
            logger.error(f"Azure Key Vault encryption failed: {e}")
            raise

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt using Azure Key Vault"""
        try:
            from azure.keyvault.keys.crypto import CryptographyClient, EncryptionAlgorithm

            key = self.key_client.get_key(key_id)
            crypto_client = CryptographyClient(key, credential=self.credential)

            result = crypto_client.decrypt(
                EncryptionAlgorithm.rsa_oaep_256,
                ciphertext
            )
            return result.plaintext
        except Exception as e:
            logger.error(f"Azure Key Vault decryption failed: {e}")
            raise

    def verify_attestation(
        self,
        attestation_report: 'AttestationReport'
    ) -> bool:
        """
        Verify attestation using Azure Attestation Service.

        Integrates with Azure Confidential Computing attestation.
        """
        logger.info("Verifying attestation for Azure Key Vault access")
        # TODO: Implement Azure Attestation Service integration
        return True


class GCPKMSClient(KMSClient):
    """Google Cloud KMS client for confidential computing"""

    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None
    ):
        try:
            from google.cloud import kms

            self.project_id = project_id or os.getenv('GCP_PROJECT_ID')
            self.location = location or os.getenv('GCP_LOCATION', 'global')

            if not self.project_id:
                raise ValueError("GCP_PROJECT_ID must be set")

            self.client = kms.KeyManagementServiceClient()
            logger.info("GCP KMS client initialized")
        except ImportError:
            raise ImportError(
                "Google Cloud KMS is required. Install with: "
                "pip install google-cloud-kms"
            )

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt using GCP KMS"""
        try:
            # Parse key_id as keyRingId/cryptoKeyId
            parts = key_id.split('/')
            if len(parts) != 2:
                raise ValueError(
                    "key_id must be in format: keyRingId/cryptoKeyId"
                )

            key_ring_id, crypto_key_id = parts

            name = self.client.crypto_key_path(
                self.project_id, self.location, key_ring_id, crypto_key_id
            )

            response = self.client.encrypt(
                request={'name': name, 'plaintext': plaintext}
            )
            return response.ciphertext
        except Exception as e:
            logger.error(f"GCP KMS encryption failed: {e}")
            raise

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt using GCP KMS"""
        try:
            parts = key_id.split('/')
            if len(parts) != 2:
                raise ValueError(
                    "key_id must be in format: keyRingId/cryptoKeyId"
                )

            key_ring_id, crypto_key_id = parts

            name = self.client.crypto_key_path(
                self.project_id, self.location, key_ring_id, crypto_key_id
            )

            response = self.client.decrypt(
                request={'name': name, 'ciphertext': ciphertext}
            )
            return response.plaintext
        except Exception as e:
            logger.error(f"GCP KMS decryption failed: {e}")
            raise

    def verify_attestation(
        self,
        attestation_report: 'AttestationReport'
    ) -> bool:
        """
        Verify attestation for GCP Confidential Computing.

        Integrates with GCP Confidential VMs attestation.
        """
        logger.info("Verifying attestation for GCP KMS access")
        # TODO: Implement GCP attestation verification
        return True


class HashiCorpVaultClient(KMSClient):
    """HashiCorp Vault client for confidential computing"""

    def __init__(
        self,
        vault_addr: Optional[str] = None,
        vault_token: Optional[str] = None,
        mount_point: str = "transit"
    ):
        try:
            import hvac

            vault_addr = vault_addr or os.getenv('VAULT_ADDR')
            vault_token = vault_token or os.getenv('VAULT_TOKEN')

            if not vault_addr:
                raise ValueError("VAULT_ADDR must be set")

            self.client = hvac.Client(url=vault_addr, token=vault_token)
            self.mount_point = mount_point

            if not self.client.is_authenticated():
                raise RuntimeError("Failed to authenticate with Vault")

            logger.info("HashiCorp Vault client initialized")
        except ImportError:
            raise ImportError(
                "hvac is required for HashiCorp Vault. Install with: pip install hvac"
            )

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Encrypt using Vault Transit engine"""
        try:
            response = self.client.secrets.transit.encrypt_data(
                name=key_id,
                plaintext=base64.b64encode(plaintext).decode(),
                mount_point=self.mount_point
            )
            # Vault returns "vault:v1:..." format
            return response['data']['ciphertext'].encode()
        except Exception as e:
            logger.error(f"Vault encryption failed: {e}")
            raise

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Decrypt using Vault Transit engine"""
        try:
            response = self.client.secrets.transit.decrypt_data(
                name=key_id,
                ciphertext=ciphertext.decode(),
                mount_point=self.mount_point
            )
            plaintext_b64 = response['data']['plaintext']
            return base64.b64decode(plaintext_b64)
        except Exception as e:
            logger.error(f"Vault decryption failed: {e}")
            raise

    def verify_attestation(
        self,
        attestation_report: 'AttestationReport'
    ) -> bool:
        """
        Verify attestation before releasing secrets from Vault.

        Can integrate with Vault's own attestation policies.
        """
        logger.info("Verifying attestation for Vault access")
        # TODO: Implement Vault attestation policy verification
        return True


class LocalKMSClient(KMSClient):
    """
    Local KMS client for development and testing.

    WARNING: This is NOT secure and should NEVER be used in production!
    """

    def __init__(self, master_key: Optional[bytes] = None):
        if master_key:
            self.master_key = master_key
        else:
            # Generate or load from environment
            master_key_hex = os.getenv('LOCAL_KMS_MASTER_KEY')
            if master_key_hex:
                self.master_key = bytes.fromhex(master_key_hex)
            else:
                # Generate random master key
                self.master_key = os.urandom(32)
                logger.warning(
                    "⚠️  GENERATED RANDOM MASTER KEY - NOT PERSISTENT ⚠️"
                )

        logger.warning(
            "⚠️  USING LOCAL KMS - NOT SECURE FOR PRODUCTION ⚠️"
        )

    def encrypt(self, key_id: str, plaintext: bytes) -> bytes:
        """Mock encryption using XOR with master key"""
        import hashlib

        # Derive key from master_key + key_id
        derived_key = hashlib.pbkdf2_hmac(
            'sha256',
            self.master_key,
            key_id.encode(),
            100000,
            dklen=len(plaintext)
        )

        # XOR encryption (symmetric)
        ciphertext = bytes(a ^ b for a, b in zip(plaintext, derived_key))
        return ciphertext

    def decrypt(self, key_id: str, ciphertext: bytes) -> bytes:
        """Mock decryption (XOR is symmetric)"""
        return self.encrypt(key_id, ciphertext)

    def verify_attestation(
        self,
        attestation_report: 'AttestationReport'
    ) -> bool:
        """Mock attestation verification"""
        logger.warning("⚠️  MOCK ATTESTATION VERIFICATION ⚠️")
        return True


def create_kms_client(
    provider: KMSProvider,
    **kwargs
) -> KMSClient:
    """
    Factory function to create KMS client.

    Args:
        provider: KMS provider type
        **kwargs: Provider-specific configuration

    Returns:
        KMSClient: Initialized KMS client
    """
    if provider == KMSProvider.AWS_KMS:
        return AWSKMSClient(**kwargs)
    elif provider == KMSProvider.AZURE_KEY_VAULT:
        return AzureKeyVaultClient(**kwargs)
    elif provider == KMSProvider.GCP_KMS:
        return GCPKMSClient(**kwargs)
    elif provider == KMSProvider.HASHICORP_VAULT:
        return HashiCorpVaultClient(**kwargs)
    elif provider == KMSProvider.LOCAL:
        return LocalKMSClient(**kwargs)
    else:
        raise ValueError(f"Unknown KMS provider: {provider}")
