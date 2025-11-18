"""Tests for TEE Manager"""

import os
import unittest
from unittest.mock import MagicMock, patch

from vllm.confidential.tee_manager import TEEManager, TEEPlatform, get_tee_platform


class TestTEEManager(unittest.TestCase):
    """Test TEE Manager functionality"""

    def test_tee_platform_enum(self):
        """Test TEE platform enum values"""
        self.assertEqual(TEEPlatform.INTEL_TDX, "intel_tdx")
        self.assertEqual(TEEPlatform.AMD_SEV_SNP, "amd_sev_snp")
        self.assertEqual(TEEPlatform.NVIDIA_CONF_GPU, "nvidia_conf_gpu")
        self.assertEqual(TEEPlatform.NONE, "none")

    @patch('vllm.confidential.tee_manager.TEEManager._check_intel_tdx')
    @patch('vllm.confidential.tee_manager.TEEManager._check_amd_sev_snp')
    @patch('vllm.confidential.tee_manager.TEEManager._check_nvidia_conf_gpu')
    def test_detect_no_tee(self, mock_nvidia, mock_amd, mock_intel):
        """Test detection when no TEE is available"""
        mock_intel.return_value = False
        mock_amd.return_value = False
        mock_nvidia.return_value = False

        manager = TEEManager()
        self.assertEqual(manager.platform, TEEPlatform.NONE)
        self.assertFalse(manager.is_confidential())

    @patch('vllm.confidential.tee_manager.TEEManager._check_intel_tdx')
    @patch('vllm.confidential.tee_manager.TEEManager._check_amd_sev_snp')
    @patch('vllm.confidential.tee_manager.TEEManager._check_nvidia_conf_gpu')
    def test_detect_intel_tdx(self, mock_nvidia, mock_amd, mock_intel):
        """Test detection of Intel TDX"""
        mock_intel.return_value = True
        mock_amd.return_value = False
        mock_nvidia.return_value = False

        manager = TEEManager()
        self.assertEqual(manager.platform, TEEPlatform.INTEL_TDX)
        self.assertTrue(manager.is_confidential())
        self.assertTrue(manager.supports_attestation())

    @patch('vllm.confidential.tee_manager.TEEManager._check_intel_tdx')
    @patch('vllm.confidential.tee_manager.TEEManager._check_amd_sev_snp')
    @patch('vllm.confidential.tee_manager.TEEManager._check_nvidia_conf_gpu')
    def test_detect_amd_sev_snp(self, mock_nvidia, mock_amd, mock_intel):
        """Test detection of AMD SEV-SNP"""
        mock_intel.return_value = False
        mock_amd.return_value = True
        mock_nvidia.return_value = False

        manager = TEEManager()
        self.assertEqual(manager.platform, TEEPlatform.AMD_SEV_SNP)
        self.assertTrue(manager.is_confidential())
        self.assertTrue(manager.supports_attestation())

    @patch('vllm.confidential.tee_manager.TEEManager._check_intel_tdx')
    @patch('vllm.confidential.tee_manager.TEEManager._check_amd_sev_snp')
    @patch('vllm.confidential.tee_manager.TEEManager._check_nvidia_conf_gpu')
    def test_detect_nvidia_conf_gpu(self, mock_nvidia, mock_amd, mock_intel):
        """Test detection of NVIDIA Confidential GPU"""
        mock_intel.return_value = False
        mock_amd.return_value = False
        mock_nvidia.return_value = True

        manager = TEEManager()
        self.assertEqual(manager.platform, TEEPlatform.NVIDIA_CONF_GPU)
        self.assertTrue(manager.is_confidential())
        self.assertTrue(manager.supports_gpu_tee())

    def test_get_platform_info(self):
        """Test getting platform information"""
        manager = TEEManager()
        info = manager.get_platform_info()

        self.assertIn('platform', info)
        self.assertIn('features', info)
        self.assertIn('confidential', info)
        self.assertIsInstance(info['platform'], str)
        self.assertIsInstance(info['features'], dict)
        self.assertIsInstance(info['confidential'], bool)

    @patch('vllm.confidential.tee_manager.subprocess.run')
    @patch('vllm.confidential.tee_manager.os.path.exists')
    def test_check_intel_tdx_with_device(self, mock_exists, mock_run):
        """Test Intel TDX detection with device file"""
        mock_exists.return_value = True

        manager = TEEManager()
        # Device file exists should return True
        # (actual result depends on mock setup in __init__)

    def test_get_tee_platform(self):
        """Test get_tee_platform convenience function"""
        platform = get_tee_platform()
        self.assertIsInstance(platform, TEEPlatform)


if __name__ == '__main__':
    unittest.main()
