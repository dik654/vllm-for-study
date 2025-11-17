"""
TEE (Trusted Execution Environment) Manager

Detects and manages TEE platforms including Intel TDX, AMD SEV-SNP,
and NVIDIA Confidential GPU.
"""

import os
import subprocess
from enum import Enum
from typing import Dict, Optional

from vllm.logger import init_logger

logger = init_logger(__name__)


class TEEPlatform(str, Enum):
    """Supported TEE platforms"""
    INTEL_TDX = "intel_tdx"
    AMD_SEV_SNP = "amd_sev_snp"
    NVIDIA_CONF_GPU = "nvidia_conf_gpu"
    NONE = "none"


class TEEManager:
    """
    Manages TEE environment detection and configuration.

    Detects available TEE technologies and provides runtime information
    for secure model execution.
    """

    def __init__(self):
        self.platform = self._detect_platform()
        self.features = self._detect_features()
        self._log_tee_status()

    def _detect_platform(self) -> TEEPlatform:
        """
        Detect which TEE platform is available.

        Returns:
            TEEPlatform: The detected TEE platform or NONE if not in TEE
        """
        # Check for Intel TDX
        if self._check_intel_tdx():
            return TEEPlatform.INTEL_TDX

        # Check for AMD SEV-SNP
        if self._check_amd_sev_snp():
            return TEEPlatform.AMD_SEV_SNP

        # Check for NVIDIA Confidential GPU
        if self._check_nvidia_conf_gpu():
            return TEEPlatform.NVIDIA_CONF_GPU

        return TEEPlatform.NONE

    def _check_intel_tdx(self) -> bool:
        """Check if running in Intel TDX environment"""
        try:
            # Check for TDX module
            if os.path.exists("/sys/module/tdx_guest"):
                return True

            # Check CPUID for TDX support
            result = subprocess.run(
                ["cpuid", "-1", "-l", "0x21"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return "TDX" in result.stdout
        except Exception as e:
            logger.debug(f"Intel TDX detection failed: {e}")
            return False

    def _check_amd_sev_snp(self) -> bool:
        """Check if running in AMD SEV-SNP environment"""
        try:
            # Check for SEV-SNP in dmesg
            if os.path.exists("/sys/kernel/mm/memory_encrypt/active"):
                with open("/sys/kernel/mm/memory_encrypt/active", "r") as f:
                    return f.read().strip() == "1"

            # Check MSR for SEV status
            result = subprocess.run(
                ["rdmsr", "0xc0010131"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and result.stdout.strip() != "0"
        except Exception as e:
            logger.debug(f"AMD SEV-SNP detection failed: {e}")
            return False

    def _check_nvidia_conf_gpu(self) -> bool:
        """Check if NVIDIA Confidential GPU is available"""
        try:
            # Check nvidia-smi for confidential computing mode
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=compute_mode,mig.mode.current",
                 "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Check for Hopper (H100/H200) or Blackwell GPUs with CC mode
                gpu_info = subprocess.run(
                    ["nvidia-smi", "--query-gpu=name",
                     "--format=csv,noheader"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )

                # Supported GPUs: Hopper (H100, H200) and Blackwell (B100, B200, RTX PRO 6000 Blackwell)
                supported_gpus = [
                    "H100", "H200",  # Hopper
                    "B100", "B200", "Blackwell", "RTX PRO 6000"  # Blackwell
                ]

                gpu_name = gpu_info.stdout.strip()
                if any(gpu in gpu_name for gpu in supported_gpus):
                    # Check for CC mode via device attributes
                    return self._check_nvidia_cc_mode()

            return False
        except Exception as e:
            logger.debug(f"NVIDIA Confidential GPU detection failed: {e}")
            return False

    def _check_nvidia_cc_mode(self) -> bool:
        """Check if NVIDIA GPU is in Confidential Computing mode"""
        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                # Check for confidential computing mode
                # Note: This requires NVML extensions for CC
                try:
                    cc_mode = pynvml.nvmlDeviceGetConfComputeMode(handle)
                    if cc_mode == pynvml.NVML_CC_ACCEPTING_CLIENT_REQUESTS_TRUE:
                        return True
                except AttributeError:
                    # CC mode API not available, check environment variable
                    if os.getenv("NVIDIA_CC_MODE") == "enabled":
                        return True

            pynvml.nvmlShutdown()
            return False
        except ImportError:
            logger.warning("pynvml not installed, cannot detect NVIDIA CC mode")
            return os.getenv("NVIDIA_CC_MODE") == "enabled"
        except Exception as e:
            logger.debug(f"NVIDIA CC mode check failed: {e}")
            return False

    def _detect_features(self) -> Dict[str, bool]:
        """
        Detect available TEE features.

        Returns:
            Dict of feature names to availability
        """
        features = {
            "memory_encryption": False,
            "remote_attestation": False,
            "gpu_tee": False,
            "secure_boot": False,
        }

        if self.platform == TEEPlatform.INTEL_TDX:
            features["memory_encryption"] = True
            features["remote_attestation"] = True
            features["secure_boot"] = True
        elif self.platform == TEEPlatform.AMD_SEV_SNP:
            features["memory_encryption"] = True
            features["remote_attestation"] = True
            features["secure_boot"] = True

        if self.platform == TEEPlatform.NVIDIA_CONF_GPU:
            features["gpu_tee"] = True
            features["memory_encryption"] = True

        return features

    def _log_tee_status(self):
        """Log detected TEE platform and features"""
        if self.platform == TEEPlatform.NONE:
            logger.warning(
                "No TEE platform detected. Running in non-confidential mode. "
                "Model weights and prompts will NOT be protected by hardware."
            )
        else:
            logger.info(f"TEE platform detected: {self.platform.value}")
            logger.info(f"TEE features: {self.features}")

    def is_confidential(self) -> bool:
        """Check if running in a confidential environment"""
        return self.platform != TEEPlatform.NONE

    def supports_gpu_tee(self) -> bool:
        """Check if GPU TEE is supported"""
        return self.features.get("gpu_tee", False)

    def supports_attestation(self) -> bool:
        """Check if remote attestation is supported"""
        return self.features.get("remote_attestation", False)

    def get_platform_info(self) -> Dict[str, any]:
        """
        Get detailed platform information.

        Returns:
            Dict with platform details
        """
        return {
            "platform": self.platform.value,
            "features": self.features,
            "confidential": self.is_confidential(),
        }


def get_tee_platform() -> TEEPlatform:
    """
    Get the current TEE platform.

    Returns:
        TEEPlatform: The detected TEE platform
    """
    manager = TEEManager()
    return manager.platform
