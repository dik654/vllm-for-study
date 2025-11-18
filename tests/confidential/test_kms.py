"""Tests for KMS functionality"""

import unittest

from vllm.confidential.kms import (
    KMSProvider,
    LocalKMSClient,
    create_kms_client,
)


class TestKMS(unittest.TestCase):
    """Test KMS client functionality"""

    def test_kms_provider_enum(self):
        """Test KMS provider enum values"""
        self.assertEqual(KMSProvider.AWS_KMS, "aws_kms")
        self.assertEqual(KMSProvider.AZURE_KEY_VAULT, "azure_key_vault")
        self.assertEqual(KMSProvider.GCP_KMS, "gcp_kms")
        self.assertEqual(KMSProvider.HASHICORP_VAULT, "hashicorp_vault")
        self.assertEqual(KMSProvider.LOCAL, "local")

    def test_local_kms_encrypt_decrypt(self):
        """Test local KMS encryption and decryption"""
        kms = LocalKMSClient()
        key_id = "test-key-123"
        plaintext = b"secret data to encrypt"

        # Encrypt
        ciphertext = kms.encrypt(key_id, plaintext)
        self.assertIsInstance(ciphertext, bytes)
        self.assertNotEqual(ciphertext, plaintext)

        # Decrypt
        decrypted = kms.decrypt(key_id, ciphertext)
        self.assertEqual(decrypted, plaintext)

    def test_local_kms_different_keys(self):
        """Test that different keys produce different ciphertext"""
        kms = LocalKMSClient()
        plaintext = b"test data"

        ciphertext1 = kms.encrypt("key1", plaintext)
        ciphertext2 = kms.encrypt("key2", plaintext)

        # Different keys should produce different ciphertext
        self.assertNotEqual(ciphertext1, ciphertext2)

        # But decryption should fail with wrong key
        # (In this simple XOR implementation, wrong key gives wrong plaintext)
        decrypted_wrong = kms.decrypt("key1", ciphertext2)
        self.assertNotEqual(decrypted_wrong, plaintext)

    def test_local_kms_with_custom_master_key(self):
        """Test local KMS with custom master key"""
        master_key = b"0" * 32  # 32 bytes master key
        kms = LocalKMSClient(master_key=master_key)

        plaintext = b"test"
        ciphertext = kms.encrypt("key", plaintext)
        decrypted = kms.decrypt("key", ciphertext)

        self.assertEqual(decrypted, plaintext)

    def test_create_kms_client_local(self):
        """Test creating local KMS client via factory"""
        kms = create_kms_client(KMSProvider.LOCAL)
        self.assertIsInstance(kms, LocalKMSClient)

    def test_kms_verify_attestation(self):
        """Test mock attestation verification"""
        from vllm.confidential.attestation import AttestationReport
        from vllm.confidential.tee_manager import TEEPlatform

        kms = LocalKMSClient()
        report = AttestationReport(
            platform=TEEPlatform.INTEL_TDX,
            report_data=b"data",
            signature=b"sig",
        )

        # Mock verification should always return True
        result = kms.verify_attestation(report)
        self.assertTrue(result)

    def test_local_kms_symmetric_encryption(self):
        """Test that encryption is deterministic for same key and plaintext"""
        kms = LocalKMSClient(master_key=b"fixed" * 6 + b"fi")  # 32 bytes

        plaintext = b"test data"
        key_id = "test-key"

        # XOR encryption is symmetric and deterministic
        ciphertext1 = kms.encrypt(key_id, plaintext)
        ciphertext2 = kms.encrypt(key_id, plaintext)

        # Should produce same result
        self.assertEqual(ciphertext1, ciphertext2)

        # Decrypting ciphertext should give plaintext
        self.assertEqual(kms.decrypt(key_id, ciphertext1), plaintext)


if __name__ == '__main__':
    unittest.main()
