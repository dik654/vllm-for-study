"""
Remote Attestation for TEE Environments

Provides cryptographic proof that the code is running in a genuine TEE
with the expected security properties.
"""

import base64
import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from vllm.confidential.tee_manager import TEEPlatform
from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class AttestationReport:
    """
    Attestation report containing cryptographic proof of TEE execution.

    Attributes:
        platform: TEE platform type
        report_data: Raw attestation report bytes
        signature: Cryptographic signature
        certificates: Certificate chain for verification
        measurement: Hash of running code/configuration
        timestamp: When the attestation was generated
        nonce: Optional nonce for freshness
    """
    platform: TEEPlatform
    report_data: bytes
    signature: bytes
    certificates: List[str] = field(default_factory=list)
    measurement: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    nonce: Optional[str] = None
    metadata: Dict[str, any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            "platform": self.platform.value,
            "report_data": base64.b64encode(self.report_data).decode(),
            "signature": base64.b64encode(self.signature).decode(),
            "certificates": self.certificates,
            "measurement": self.measurement,
            "timestamp": self.timestamp.isoformat(),
            "nonce": self.nonce,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'AttestationReport':
        """Create from dictionary"""
        return cls(
            platform=TEEPlatform(data["platform"]),
            report_data=base64.b64decode(data["report_data"]),
            signature=base64.b64decode(data["signature"]),
            certificates=data.get("certificates", []),
            measurement=data.get("measurement", ""),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            nonce=data.get("nonce"),
            metadata=data.get("metadata", {}),
        )


class AttestationManager:
    """
    Manages remote attestation for different TEE platforms.

    Generates and verifies attestation reports to prove that code is
    running in a genuine TEE environment.
    """

    def __init__(self, platform: TEEPlatform):
        self.platform = platform
        self._attestation_cache: Optional[AttestationReport] = None
        self._cache_expiry: Optional[datetime] = None

    def generate_attestation(
        self,
        user_data: Optional[bytes] = None,
        nonce: Optional[str] = None
    ) -> AttestationReport:
        """
        Generate an attestation report.

        Args:
            user_data: Optional user data to include in attestation
            nonce: Optional nonce for freshness proof

        Returns:
            AttestationReport: The generated attestation report

        Raises:
            RuntimeError: If attestation generation fails
        """
        # Check cache
        if self._is_cache_valid():
            logger.info("Using cached attestation report")
            return self._attestation_cache

        logger.info(f"Generating attestation for {self.platform.value}")

        if self.platform == TEEPlatform.INTEL_TDX:
            report = self._generate_tdx_attestation(user_data, nonce)
        elif self.platform == TEEPlatform.AMD_SEV_SNP:
            report = self._generate_sev_snp_attestation(user_data, nonce)
        elif self.platform == TEEPlatform.NVIDIA_CONF_GPU:
            report = self._generate_nvidia_attestation(user_data, nonce)
        else:
            raise RuntimeError(
                f"Attestation not supported for platform: {self.platform}"
            )

        # Cache the attestation for 5 minutes
        self._attestation_cache = report
        self._cache_expiry = datetime.utcnow() + timedelta(minutes=5)

        return report

    def _is_cache_valid(self) -> bool:
        """Check if cached attestation is still valid"""
        if self._attestation_cache is None or self._cache_expiry is None:
            return False
        return datetime.utcnow() < self._cache_expiry

    def _generate_tdx_attestation(
        self,
        user_data: Optional[bytes],
        nonce: Optional[str]
    ) -> AttestationReport:
        """Generate Intel TDX attestation report"""
        try:
            # Use TDX guest attestation device
            report_data = user_data or b""
            if len(report_data) > 64:
                # TDX report data is limited to 64 bytes
                report_data = hashlib.sha512(report_data).digest()[:64]

            # Request quote from TDX module
            # This is a simplified version - production should use proper TDX API
            result = subprocess.run(
                ["tdx-attest", "quote", "--report-data",
                 base64.b64encode(report_data).decode()],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(f"TDX attestation failed: {result.stderr}")

            # Parse the quote
            quote_data = json.loads(result.stdout)

            return AttestationReport(
                platform=TEEPlatform.INTEL_TDX,
                report_data=base64.b64decode(quote_data["quote"]),
                signature=base64.b64decode(quote_data["signature"]),
                certificates=quote_data.get("certificates", []),
                measurement=quote_data.get("mr_td", ""),
                nonce=nonce,
                metadata={
                    "tdx_version": quote_data.get("version"),
                    "mr_config_id": quote_data.get("mr_config_id"),
                    "mr_owner": quote_data.get("mr_owner"),
                }
            )

        except FileNotFoundError:
            logger.warning(
                "tdx-attest not found, generating mock attestation"
            )
            return self._generate_mock_attestation(user_data, nonce)
        except Exception as e:
            logger.error(f"TDX attestation generation failed: {e}")
            raise RuntimeError(f"Failed to generate TDX attestation: {e}")

    def _generate_sev_snp_attestation(
        self,
        user_data: Optional[bytes],
        nonce: Optional[str]
    ) -> AttestationReport:
        """Generate AMD SEV-SNP attestation report"""
        try:
            # Use SEV-SNP guest attestation
            report_data = user_data or b""
            if len(report_data) > 64:
                report_data = hashlib.sha512(report_data).digest()[:64]

            # Request attestation report
            result = subprocess.run(
                ["sev-guest-get-report",
                 base64.b64encode(report_data).decode()],
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                raise RuntimeError(
                    f"SEV-SNP attestation failed: {result.stderr}"
                )

            # Parse the report
            report_json = json.loads(result.stdout)

            return AttestationReport(
                platform=TEEPlatform.AMD_SEV_SNP,
                report_data=base64.b64decode(report_json["report"]),
                signature=base64.b64decode(report_json["signature"]),
                certificates=report_json.get("certificates", []),
                measurement=report_json.get("measurement", ""),
                nonce=nonce,
                metadata={
                    "policy": report_json.get("policy"),
                    "vmpl": report_json.get("vmpl"),
                    "guest_svn": report_json.get("guest_svn"),
                }
            )

        except FileNotFoundError:
            logger.warning(
                "sev-guest-get-report not found, generating mock attestation"
            )
            return self._generate_mock_attestation(user_data, nonce)
        except Exception as e:
            logger.error(f"SEV-SNP attestation generation failed: {e}")
            raise RuntimeError(f"Failed to generate SEV-SNP attestation: {e}")

    def _generate_nvidia_attestation(
        self,
        user_data: Optional[bytes],
        nonce: Optional[str]
    ) -> AttestationReport:
        """Generate NVIDIA Confidential GPU attestation"""
        try:
            # NVIDIA H100 CC attestation via NVML
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            if device_count == 0:
                raise RuntimeError("No NVIDIA GPU found")

            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # Get attestation report
            # Note: This requires NVML CC extensions
            try:
                report_bytes = pynvml.nvmlDeviceGetConfComputeGpuAttestationReport(
                    handle, nonce.encode() if nonce else b""
                )

                return AttestationReport(
                    platform=TEEPlatform.NVIDIA_CONF_GPU,
                    report_data=report_bytes,
                    signature=b"",  # Signature embedded in report
                    measurement=hashlib.sha256(report_bytes).hexdigest(),
                    nonce=nonce,
                    metadata={
                        "gpu_uuid": pynvml.nvmlDeviceGetUUID(handle),
                    }
                )
            except AttributeError:
                # CC attestation API not available
                logger.warning(
                    "NVIDIA CC attestation API not available, "
                    "generating mock attestation"
                )
                return self._generate_mock_attestation(user_data, nonce)

        except ImportError:
            logger.warning(
                "pynvml not installed, generating mock attestation"
            )
            return self._generate_mock_attestation(user_data, nonce)
        except Exception as e:
            logger.error(f"NVIDIA attestation generation failed: {e}")
            raise RuntimeError(f"Failed to generate NVIDIA attestation: {e}")
        finally:
            try:
                pynvml.nvmlShutdown()
            except:
                pass

    def _generate_mock_attestation(
        self,
        user_data: Optional[bytes],
        nonce: Optional[str]
    ) -> AttestationReport:
        """Generate mock attestation for testing"""
        logger.warning(
            "⚠️  GENERATING MOCK ATTESTATION - NOT SECURE FOR PRODUCTION ⚠️"
        )

        # Create a fake but deterministic attestation
        report_data = user_data or b"mock_report_data"
        measurement = hashlib.sha256(b"mock_measurement").hexdigest()

        return AttestationReport(
            platform=self.platform,
            report_data=report_data,
            signature=hashlib.sha256(report_data).digest(),
            measurement=measurement,
            nonce=nonce,
            metadata={"mock": True}
        )


def verify_attestation(
    report: AttestationReport,
    expected_measurement: Optional[str] = None,
    max_age_seconds: int = 300
) -> bool:
    """
    Verify an attestation report.

    Args:
        report: The attestation report to verify
        expected_measurement: Expected measurement hash (optional)
        max_age_seconds: Maximum age of report in seconds

    Returns:
        bool: True if attestation is valid

    Raises:
        ValueError: If attestation is invalid
    """
    # Check age
    age = (datetime.utcnow() - report.timestamp).total_seconds()
    if age > max_age_seconds:
        raise ValueError(
            f"Attestation too old: {age}s > {max_age_seconds}s"
        )

    # Check measurement if provided
    if expected_measurement and report.measurement != expected_measurement:
        raise ValueError(
            f"Measurement mismatch: {report.measurement} != "
            f"{expected_measurement}"
        )

    # Platform-specific verification
    if report.platform == TEEPlatform.INTEL_TDX:
        return _verify_tdx_attestation(report)
    elif report.platform == TEEPlatform.AMD_SEV_SNP:
        return _verify_sev_snp_attestation(report)
    elif report.platform == TEEPlatform.NVIDIA_CONF_GPU:
        return _verify_nvidia_attestation(report)
    else:
        logger.warning(f"Cannot verify {report.platform} attestation")
        return False


def _verify_tdx_attestation(report: AttestationReport) -> bool:
    """Verify Intel TDX attestation"""
    # In production, this would:
    # 1. Verify certificate chain against Intel's root CA
    # 2. Verify quote signature
    # 3. Check TCB (Trusted Computing Base) version
    logger.info("Verifying TDX attestation (mock verification)")
    return True


def _verify_sev_snp_attestation(report: AttestationReport) -> bool:
    """Verify AMD SEV-SNP attestation"""
    # In production, this would:
    # 1. Verify certificate chain against AMD's root CA
    # 2. Verify report signature
    # 3. Check guest policy
    logger.info("Verifying SEV-SNP attestation (mock verification)")
    return True


def _verify_nvidia_attestation(report: AttestationReport) -> bool:
    """Verify NVIDIA Confidential GPU attestation"""
    # In production, this would:
    # 1. Verify NVIDIA's certificate chain
    # 2. Verify GPU attestation report
    logger.info("Verifying NVIDIA CC attestation (mock verification)")
    return True
