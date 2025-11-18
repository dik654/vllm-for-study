"""Tests for Encryption functionality"""

import os
import tempfile
import unittest
from pathlib import Path

from vllm.confidential.encryption import (
    EncryptionManager,
    EncryptionMetadata,
    ModelEncryption,
    decrypt_model_weights,
    encrypt_model_weights,
)
from vllm.confidential.kms import LocalKMSClient


class TestEncryption(unittest.TestCase):
    """Test encryption functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.key_id = "test-key-id"
        self.kms_client = LocalKMSClient()

    def tearDown(self):
        """Clean up test files"""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_encryption_metadata_to_dict(self):
        """Test EncryptionMetadata serialization"""
        metadata = EncryptionMetadata(
            algorithm="AES-256-GCM",
            key_id="test-key",
            encrypted_dek=b"encrypted_dek",
            iv=b"initialization_vector",
            auth_tag=b"auth_tag",
            model_hash="abc123",
        )

        data = metadata.to_dict()
        self.assertEqual(data['algorithm'], "AES-256-GCM")
        self.assertEqual(data['key_id'], "test-key")
        self.assertIn('encrypted_dek', data)
        self.assertIn('iv', data)

    def test_encryption_metadata_from_dict(self):
        """Test EncryptionMetadata deserialization"""
        import base64

        data = {
            "algorithm": "AES-256-GCM",
            "key_id": "test-key",
            "encrypted_dek": base64.b64encode(b"encrypted_dek").decode(),
            "iv": base64.b64encode(b"iv").decode(),
            "auth_tag": base64.b64encode(b"tag").decode(),
            "model_hash": "abc123",
            "encrypted_files": {}
        }

        metadata = EncryptionMetadata.from_dict(data)
        self.assertEqual(metadata.algorithm, "AES-256-GCM")
        self.assertEqual(metadata.key_id, "test-key")
        self.assertEqual(metadata.encrypted_dek, b"encrypted_dek")

    def test_encrypt_decrypt_file(self):
        """Test file encryption and decryption"""
        # Create test file
        test_file = Path(self.temp_dir) / "test.txt"
        test_content = b"This is a test file for encryption"
        with open(test_file, 'wb') as f:
            f.write(test_content)

        # Encrypt
        encrypted_file = Path(self.temp_dir) / "test.txt.enc"
        encryptor = ModelEncryption(self.key_id, self.kms_client)
        metadata = encryptor.encrypt_file(test_file, encrypted_file)

        self.assertTrue(encrypted_file.exists())
        self.assertEqual(metadata.algorithm, "AES-256-GCM")
        self.assertEqual(metadata.key_id, self.key_id)

        # Decrypt
        decrypted_file = Path(self.temp_dir) / "test_decrypted.txt"
        success = encryptor.decrypt_file(encrypted_file, decrypted_file, metadata)

        self.assertTrue(success)
        self.assertTrue(decrypted_file.exists())

        # Verify content
        with open(decrypted_file, 'rb') as f:
            decrypted_content = f.read()

        self.assertEqual(decrypted_content, test_content)

    def test_encrypt_decrypt_model(self):
        """Test model directory encryption and decryption"""
        # Create test model directory
        model_dir = Path(self.temp_dir) / "model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text('{"model": "test"}')
        (model_dir / "weights.bin").write_bytes(b"fake_weights" * 100)

        # Encrypt model
        encrypted_dir = Path(self.temp_dir) / "encrypted_model"
        manager = EncryptionManager(self.key_id, self.kms_client)
        metadata = manager.encrypt_model(model_dir, encrypted_dir)

        self.assertTrue(encrypted_dir.exists())
        self.assertTrue((encrypted_dir / "encryption_metadata.json").exists())

        # Decrypt model
        decrypted_dir = Path(self.temp_dir) / "decrypted_model"
        success = manager.decrypt_model(encrypted_dir, decrypted_dir)

        self.assertTrue(success)
        self.assertTrue(decrypted_dir.exists())
        self.assertTrue((decrypted_dir / "config.json").exists())
        self.assertTrue((decrypted_dir / "weights.bin").exists())

        # Verify content
        self.assertEqual(
            (model_dir / "config.json").read_text(),
            (decrypted_dir / "config.json").read_text()
        )
        self.assertEqual(
            (model_dir / "weights.bin").read_bytes(),
            (decrypted_dir / "weights.bin").read_bytes()
        )

    def test_encrypt_model_weights_convenience(self):
        """Test convenience functions for encryption"""
        # Create test model
        model_dir = Path(self.temp_dir) / "model"
        model_dir.mkdir()
        (model_dir / "test.txt").write_text("test")

        encrypted_dir = Path(self.temp_dir) / "encrypted"
        metadata = encrypt_model_weights(
            model_dir,
            encrypted_dir,
            self.key_id,
            self.kms_client
        )

        self.assertTrue(encrypted_dir.exists())
        self.assertIsInstance(metadata, EncryptionMetadata)

    def test_hash_calculation(self):
        """Test file hash calculation"""
        test_file = Path(self.temp_dir) / "test.bin"
        test_content = b"test content for hashing"
        test_file.write_bytes(test_content)

        encryptor = ModelEncryption(self.key_id, self.kms_client)
        file_hash = encryptor._calculate_file_hash(test_file)

        import hashlib
        expected_hash = hashlib.sha256(test_content).hexdigest()
        self.assertEqual(file_hash, expected_hash)


if __name__ == '__main__':
    unittest.main()
