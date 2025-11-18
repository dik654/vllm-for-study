"""Tests for Attestation functionality"""

import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from vllm.confidential.attestation import (
    AttestationManager,
    AttestationReport,
    verify_attestation,
)
from vllm.confidential.tee_manager import TEEPlatform


class TestAttestation(unittest.TestCase):
    """Test attestation functionality"""

    def test_attestation_report_to_dict(self):
        """Test AttestationReport serialization"""
        report = AttestationReport(
            platform=TEEPlatform.INTEL_TDX,
            report_data=b"report_data",
            signature=b"signature",
            certificates=["cert1", "cert2"],
            measurement="abc123",
            nonce="test_nonce",
        )

        data = report.to_dict()
        self.assertEqual(data['platform'], "intel_tdx")
        self.assertEqual(data['measurement'], "abc123")
        self.assertEqual(data['nonce'], "test_nonce")
        self.assertEqual(len(data['certificates']), 2)

    def test_attestation_report_from_dict(self):
        """Test AttestationReport deserialization"""
        import base64

        data = {
            "platform": "intel_tdx",
            "report_data": base64.b64encode(b"data").decode(),
            "signature": base64.b64encode(b"sig").decode(),
            "certificates": ["cert1"],
            "measurement": "abc123",
            "timestamp": datetime.utcnow().isoformat(),
            "nonce": "test",
            "metadata": {"test": "value"}
        }

        report = AttestationReport.from_dict(data)
        self.assertEqual(report.platform, TEEPlatform.INTEL_TDX)
        self.assertEqual(report.measurement, "abc123")
        self.assertEqual(report.nonce, "test")

    @patch('vllm.confidential.attestation.subprocess.run')
    def test_generate_mock_attestation(self, mock_run):
        """Test mock attestation generation"""
        # Simulate tdx-attest not found
        mock_run.side_effect = FileNotFoundError()

        manager = AttestationManager(TEEPlatform.INTEL_TDX)
        report = manager.generate_attestation(nonce="test123")

        self.assertIsInstance(report, AttestationReport)
        self.assertEqual(report.platform, TEEPlatform.INTEL_TDX)
        self.assertEqual(report.nonce, "test123")
        self.assertTrue(report.metadata.get('mock'))

    def test_attestation_caching(self):
        """Test attestation report caching"""
        manager = AttestationManager(TEEPlatform.INTEL_TDX)

        # Generate first attestation
        report1 = manager.generate_attestation()

        # Generate second attestation (should be cached)
        report2 = manager.generate_attestation()

        # Should return same cached report
        self.assertEqual(report1.timestamp, report2.timestamp)

    def test_verify_attestation_age(self):
        """Test attestation age verification"""
        # Create old attestation
        old_report = AttestationReport(
            platform=TEEPlatform.INTEL_TDX,
            report_data=b"data",
            signature=b"sig",
            timestamp=datetime.utcnow() - timedelta(minutes=10)
        )

        # Should fail due to age
        with self.assertRaises(ValueError):
            verify_attestation(old_report, max_age_seconds=300)

    def test_verify_attestation_measurement(self):
        """Test attestation measurement verification"""
        report = AttestationReport(
            platform=TEEPlatform.INTEL_TDX,
            report_data=b"data",
            signature=b"sig",
            measurement="abc123",
        )

        # Should fail with wrong measurement
        with self.assertRaises(ValueError):
            verify_attestation(report, expected_measurement="xyz789")

        # Should succeed with correct measurement
        result = verify_attestation(report, expected_measurement="abc123")
        self.assertTrue(result)

    def test_verify_attestation_fresh(self):
        """Test verification of fresh attestation"""
        report = AttestationReport(
            platform=TEEPlatform.INTEL_TDX,
            report_data=b"data",
            signature=b"sig",
            measurement="abc123",
            timestamp=datetime.utcnow()
        )

        # Should succeed
        result = verify_attestation(report, max_age_seconds=300)
        self.assertTrue(result)

    def test_different_platforms(self):
        """Test attestation for different platforms"""
        platforms = [
            TEEPlatform.INTEL_TDX,
            TEEPlatform.AMD_SEV_SNP,
            TEEPlatform.NVIDIA_CONF_GPU,
        ]

        for platform in platforms:
            manager = AttestationManager(platform)
            report = manager.generate_attestation()

            self.assertEqual(report.platform, platform)
            self.assertIsInstance(report, AttestationReport)


if __name__ == '__main__':
    unittest.main()
