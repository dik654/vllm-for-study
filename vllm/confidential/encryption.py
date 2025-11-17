"""
Encryption Manager for Model Weights

Provides secure encryption and decryption of model weights using
envelope encryption with KMS integration.
"""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Optional, Union

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class EncryptionMetadata:
    """
    Metadata for encrypted model weights.

    Attributes:
        algorithm: Encryption algorithm used (e.g., "AES-256-GCM")
        key_id: KMS key ID used for envelope encryption
        encrypted_dek: Encrypted data encryption key
        iv: Initialization vector
        auth_tag: Authentication tag for GCM mode
        model_hash: Hash of original unencrypted model
        encrypted_files: List of encrypted file paths
    """
    algorithm: str
    key_id: str
    encrypted_dek: bytes
    iv: bytes
    auth_tag: Optional[bytes] = None
    model_hash: Optional[str] = None
    encrypted_files: Dict[str, str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        import base64
        return {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
            "encrypted_dek": base64.b64encode(self.encrypted_dek).decode(),
            "iv": base64.b64encode(self.iv).decode(),
            "auth_tag": base64.b64encode(self.auth_tag).decode() if self.auth_tag else None,
            "model_hash": self.model_hash,
            "encrypted_files": self.encrypted_files or {},
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'EncryptionMetadata':
        """Create from dictionary"""
        import base64
        return cls(
            algorithm=data["algorithm"],
            key_id=data["key_id"],
            encrypted_dek=base64.b64decode(data["encrypted_dek"]),
            iv=base64.b64decode(data["iv"]),
            auth_tag=base64.b64decode(data["auth_tag"]) if data.get("auth_tag") else None,
            model_hash=data.get("model_hash"),
            encrypted_files=data.get("encrypted_files", {}),
        )


class ModelEncryption:
    """
    Handles encryption and decryption of model weights.

    Uses AES-256-GCM for encryption with envelope encryption pattern:
    1. Generate a random Data Encryption Key (DEK)
    2. Encrypt model weights with DEK
    3. Encrypt DEK with KMS Key Encryption Key (KEK)
    4. Store encrypted DEK with encrypted model
    """

    def __init__(
        self,
        key_id: str,
        kms_client: Optional['KMSClient'] = None
    ):
        """
        Initialize model encryption.

        Args:
            key_id: KMS key ID for envelope encryption
            kms_client: KMS client for key operations
        """
        self.key_id = key_id
        self.kms_client = kms_client
        self.algorithm = "AES-256-GCM"

    def encrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> EncryptionMetadata:
        """
        Encrypt a single model file.

        Args:
            input_path: Path to unencrypted file
            output_path: Path for encrypted file

        Returns:
            EncryptionMetadata: Encryption metadata
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Encrypting {input_path} -> {output_path}")

        # Generate random DEK (32 bytes for AES-256)
        dek = os.urandom(32)

        # Generate random IV (12 bytes for GCM)
        iv = os.urandom(12)

        # Calculate hash of original file
        file_hash = self._calculate_file_hash(input_path)

        # Encrypt file with DEK
        auth_tag = self._encrypt_file_with_key(
            input_path, output_path, dek, iv
        )

        # Encrypt DEK with KMS
        encrypted_dek = self._encrypt_dek(dek)

        # Securely erase DEK from memory
        del dek

        return EncryptionMetadata(
            algorithm=self.algorithm,
            key_id=self.key_id,
            encrypted_dek=encrypted_dek,
            iv=iv,
            auth_tag=auth_tag,
            model_hash=file_hash,
        )

    def decrypt_file(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        metadata: EncryptionMetadata
    ) -> bool:
        """
        Decrypt a model file.

        Args:
            input_path: Path to encrypted file
            output_path: Path for decrypted file
            metadata: Encryption metadata

        Returns:
            bool: True if decryption successful
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        logger.info(f"Decrypting {input_path} -> {output_path}")

        # Decrypt DEK using KMS
        dek = self._decrypt_dek(metadata.encrypted_dek)

        try:
            # Decrypt file with DEK
            self._decrypt_file_with_key(
                input_path, output_path, dek, metadata.iv, metadata.auth_tag
            )

            # Verify hash if available
            if metadata.model_hash:
                actual_hash = self._calculate_file_hash(output_path)
                if actual_hash != metadata.model_hash:
                    logger.error("Decrypted file hash mismatch!")
                    return False

            return True
        finally:
            # Securely erase DEK from memory
            del dek

    def _encrypt_file_with_key(
        self,
        input_path: Path,
        output_path: Path,
        key: bytes,
        iv: bytes
    ) -> bytes:
        """Encrypt file with AES-256-GCM"""
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv),
            backend=default_backend()
        )
        encryptor = cipher.encryptor()

        # Encrypt file in chunks
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            while True:
                chunk = fin.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                encrypted_chunk = encryptor.update(chunk)
                fout.write(encrypted_chunk)

        # Finalize and get auth tag
        encryptor.finalize()
        return encryptor.tag

    def _decrypt_file_with_key(
        self,
        input_path: Path,
        output_path: Path,
        key: bytes,
        iv: bytes,
        auth_tag: Optional[bytes]
    ):
        """Decrypt file with AES-256-GCM"""
        # Create cipher
        cipher = Cipher(
            algorithms.AES(key),
            modes.GCM(iv, auth_tag),
            backend=default_backend()
        )
        decryptor = cipher.decryptor()

        # Decrypt file in chunks
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(input_path, 'rb') as fin, open(output_path, 'wb') as fout:
            while True:
                chunk = fin.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                decrypted_chunk = decryptor.update(chunk)
                fout.write(decrypted_chunk)

        # Finalize (will raise error if auth tag invalid)
        decryptor.finalize()

    def _encrypt_dek(self, dek: bytes) -> bytes:
        """
        Encrypt DEK using KMS.

        Args:
            dek: Data encryption key to encrypt

        Returns:
            bytes: Encrypted DEK
        """
        if self.kms_client:
            return self.kms_client.encrypt(self.key_id, dek)
        else:
            # For development/testing: use simple encryption
            # In production, this MUST use a real KMS
            logger.warning(
                "⚠️  NO KMS CLIENT - Using mock encryption (NOT SECURE) ⚠️"
            )
            return self._mock_encrypt_dek(dek)

    def _decrypt_dek(self, encrypted_dek: bytes) -> bytes:
        """
        Decrypt DEK using KMS.

        Args:
            encrypted_dek: Encrypted data encryption key

        Returns:
            bytes: Decrypted DEK
        """
        if self.kms_client:
            return self.kms_client.decrypt(self.key_id, encrypted_dek)
        else:
            # For development/testing
            logger.warning(
                "⚠️  NO KMS CLIENT - Using mock decryption (NOT SECURE) ⚠️"
            )
            return self._mock_decrypt_dek(encrypted_dek)

    def _mock_encrypt_dek(self, dek: bytes) -> bytes:
        """Mock DEK encryption for testing"""
        # Simple XOR with key_id hash (NOT SECURE)
        key = hashlib.sha256(self.key_id.encode()).digest()
        return bytes(a ^ b for a, b in zip(dek, key))

    def _mock_decrypt_dek(self, encrypted_dek: bytes) -> bytes:
        """Mock DEK decryption for testing"""
        # XOR is symmetric
        return self._mock_encrypt_dek(encrypted_dek)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                chunk = f.read(1024 * 1024)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()


class EncryptionManager:
    """
    High-level manager for model encryption operations.

    Handles encryption of entire model directories and metadata management.
    """

    def __init__(
        self,
        key_id: str,
        kms_client: Optional['KMSClient'] = None
    ):
        self.key_id = key_id
        self.kms_client = kms_client

    def encrypt_model(
        self,
        model_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> EncryptionMetadata:
        """
        Encrypt entire model directory.

        Args:
            model_path: Path to model directory
            output_path: Path for encrypted model

        Returns:
            EncryptionMetadata: Combined encryption metadata
        """
        import json

        model_path = Path(model_path)
        output_path = Path(output_path)

        logger.info(f"Encrypting model: {model_path} -> {output_path}")

        # Create output directory
        output_path.mkdir(parents=True, exist_ok=True)

        # Encrypt all model files
        encryptor = ModelEncryption(self.key_id, self.kms_client)
        encrypted_files = {}

        for file_path in model_path.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(model_path)
                encrypted_file_path = output_path / f"{rel_path}.enc"

                metadata = encryptor.encrypt_file(file_path, encrypted_file_path)
                encrypted_files[str(rel_path)] = metadata.to_dict()

        # Save combined metadata
        combined_metadata = EncryptionMetadata(
            algorithm="AES-256-GCM",
            key_id=self.key_id,
            encrypted_dek=b"",  # Per-file DEKs stored in encrypted_files
            iv=b"",
            encrypted_files=encrypted_files,
        )

        metadata_path = output_path / "encryption_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(combined_metadata.to_dict(), f, indent=2)

        logger.info(f"Model encrypted successfully: {len(encrypted_files)} files")
        return combined_metadata

    def decrypt_model(
        self,
        encrypted_path: Union[str, Path],
        output_path: Union[str, Path]
    ) -> bool:
        """
        Decrypt entire model directory.

        Args:
            encrypted_path: Path to encrypted model
            output_path: Path for decrypted model

        Returns:
            bool: True if successful
        """
        import json

        encrypted_path = Path(encrypted_path)
        output_path = Path(output_path)

        logger.info(f"Decrypting model: {encrypted_path} -> {output_path}")

        # Load metadata
        metadata_path = encrypted_path / "encryption_metadata.json"
        with open(metadata_path, 'r') as f:
            combined_data = json.load(f)

        combined_metadata = EncryptionMetadata.from_dict(combined_data)

        # Decrypt all files
        decryptor = ModelEncryption(self.key_id, self.kms_client)
        output_path.mkdir(parents=True, exist_ok=True)

        for rel_path, file_metadata_dict in combined_metadata.encrypted_files.items():
            file_metadata = EncryptionMetadata.from_dict(file_metadata_dict)

            encrypted_file = encrypted_path / f"{rel_path}.enc"
            decrypted_file = output_path / rel_path

            decrypted_file.parent.mkdir(parents=True, exist_ok=True)

            success = decryptor.decrypt_file(
                encrypted_file, decrypted_file, file_metadata
            )

            if not success:
                logger.error(f"Failed to decrypt {rel_path}")
                return False

        logger.info("Model decrypted successfully")
        return True


def encrypt_model_weights(
    model_path: Union[str, Path],
    output_path: Union[str, Path],
    key_id: str,
    kms_client: Optional['KMSClient'] = None
) -> EncryptionMetadata:
    """
    Convenience function to encrypt model weights.

    Args:
        model_path: Path to unencrypted model
        output_path: Path for encrypted model
        key_id: KMS key ID
        kms_client: Optional KMS client

    Returns:
        EncryptionMetadata: Encryption metadata
    """
    manager = EncryptionManager(key_id, kms_client)
    return manager.encrypt_model(model_path, output_path)


def decrypt_model_weights(
    encrypted_path: Union[str, Path],
    output_path: Union[str, Path],
    key_id: str,
    kms_client: Optional['KMSClient'] = None
) -> bool:
    """
    Convenience function to decrypt model weights.

    Args:
        encrypted_path: Path to encrypted model
        output_path: Path for decrypted model
        key_id: KMS key ID
        kms_client: Optional KMS client

    Returns:
        bool: True if successful
    """
    manager = EncryptionManager(key_id, kms_client)
    return manager.decrypt_model(encrypted_path, output_path)
