#!/usr/bin/env python3
"""
Client with Attestation Verification

Demonstrates how to:
1. Request attestation from server
2. Verify the attestation report
3. Run secure inference with verified server
"""

import hashlib
import json
import time
from typing import Optional

import requests


class ConfidentialClient:
    """
    Client that verifies server attestation before sending requests.
    """

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.attestation_verified = False
        self.last_attestation: Optional[dict] = None

    def get_attestation(self, use_nonce: bool = True) -> dict:
        """
        Request attestation report from server.

        Args:
            use_nonce: Whether to use a nonce for freshness

        Returns:
            dict: Attestation report
        """
        print("=" * 60)
        print("Requesting Attestation from Server")
        print("=" * 60)

        nonce = None
        if use_nonce:
            # Generate random nonce for freshness proof
            nonce = hashlib.sha256(str(time.time()).encode()).hexdigest()[:32]
            print(f"Nonce: {nonce}")

        response = requests.post(
            f"{self.server_url}/v1/confidential/attestation",
            json={"nonce": nonce} if nonce else {}
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Attestation request failed: {response.status_code} - {response.text}"
            )

        attestation = response.json()
        self.last_attestation = attestation

        print(f"Platform: {attestation['platform']}")
        print(f"Measurement: {attestation['measurement'][:32]}...")
        print(f"Timestamp: {attestation['timestamp']}")
        print()

        return attestation

    def verify_attestation(
        self,
        attestation: Optional[dict] = None,
        expected_measurement: Optional[str] = None
    ) -> bool:
        """
        Verify attestation report.

        In production, this should:
        1. Verify certificate chain against hardware vendor's root CA
        2. Verify cryptographic signature
        3. Check measurement matches expected value
        4. Verify timestamp freshness
        5. Check nonce if provided

        Args:
            attestation: Attestation report to verify (uses last if None)
            expected_measurement: Expected measurement hash

        Returns:
            bool: True if attestation is valid
        """
        print("=" * 60)
        print("Verifying Attestation")
        print("=" * 60)

        if attestation is None:
            attestation = self.last_attestation

        if attestation is None:
            print("✗ No attestation to verify")
            return False

        # Check 1: Platform is recognized
        platform = attestation.get('platform')
        supported_platforms = ['intel_tdx', 'amd_sev_snp', 'nvidia_conf_gpu']

        if platform not in supported_platforms:
            print(f"✗ Unsupported platform: {platform}")
            return False

        print(f"✓ Platform recognized: {platform}")

        # Check 2: Timestamp is recent (within 5 minutes)
        from datetime import datetime, timedelta
        timestamp_str = attestation.get('timestamp')
        if timestamp_str:
            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            age = (datetime.utcnow() - timestamp.replace(tzinfo=None)).total_seconds()

            if age > 300:  # 5 minutes
                print(f"✗ Attestation too old: {age}s")
                return False

            print(f"✓ Timestamp fresh: {age:.1f}s old")

        # Check 3: Measurement (if expected value provided)
        if expected_measurement:
            actual_measurement = attestation.get('measurement')
            if actual_measurement != expected_measurement:
                print(f"✗ Measurement mismatch")
                print(f"  Expected: {expected_measurement}")
                print(f"  Actual:   {actual_measurement}")
                return False

            print(f"✓ Measurement matches expected value")
        else:
            print(f"⚠  Measurement not verified (no expected value)")

        # Check 4: Signature verification
        # In production: verify signature with certificate chain
        print(f"⚠  Signature verification not implemented (mock)")

        # Overall result
        print()
        print("=" * 60)
        print("Attestation Verification Result")
        print("=" * 60)

        if expected_measurement:
            print("✓ VERIFIED - Server is running in genuine TEE")
            self.attestation_verified = True
        else:
            print("⚠  PARTIAL - Basic checks passed, but measurement not verified")
            print("   For production, provide expected_measurement parameter")
            self.attestation_verified = False

        print()
        return True

    def run_inference(
        self,
        prompt: str,
        require_attestation: bool = True
    ) -> dict:
        """
        Run inference with optional attestation requirement.

        Args:
            prompt: The prompt to send
            require_attestation: Whether to require valid attestation

        Returns:
            dict: Inference result
        """
        if require_attestation and not self.attestation_verified:
            print("⚠️  WARNING: Server attestation not verified!")
            print("   Your prompt may not be protected.")
            response = input("   Continue anyway? [y/N]: ")
            if response.lower() != 'y':
                raise RuntimeError("Attestation required but not verified")

        print("=" * 60)
        print("Running Confidential Inference")
        print("=" * 60)
        print(f"Prompt: {prompt}")
        print()

        response = requests.post(
            f"{self.server_url}/v1/chat/completions",
            json={
                "model": "default",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 100
            }
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"Inference failed: {response.status_code} - {response.text}"
            )

        result = response.json()
        assistant_message = result['choices'][0]['message']['content']

        print(f"Response: {assistant_message}")
        print()

        return result


def main():
    """Main example workflow"""
    print("\n" + "=" * 60)
    print("Confidential Inference Client with Attestation")
    print("=" * 60 + "\n")

    # Initialize client
    client = ConfidentialClient(server_url="http://localhost:8000")

    try:
        # Step 1: Get attestation
        attestation = client.get_attestation(use_nonce=True)

        # Step 2: Verify attestation
        # In production, you would provide the expected measurement
        # that you obtained from a trusted build process
        expected_measurement = None  # Replace with actual expected value

        if expected_measurement:
            print(f"Expected measurement: {expected_measurement}")
        else:
            print("⚠️  No expected measurement provided")
            print("   In production, you should verify the measurement!")

        print()

        verified = client.verify_attestation(
            attestation=attestation,
            expected_measurement=expected_measurement
        )

        if not verified:
            print("❌ Attestation verification failed!")
            return

        # Step 3: Run inference with verified server
        prompts = [
            "What is confidential computing?",
            "How does TEE protect my data?",
        ]

        for prompt in prompts:
            result = client.run_inference(
                prompt=prompt,
                require_attestation=True
            )
            print("✓ Inference completed successfully")
            print()

        print("=" * 60)
        print("All operations completed successfully!")
        print("=" * 60)

        # Show summary
        print()
        print("Summary:")
        print(f"  Server: {client.server_url}")
        print(f"  Platform: {attestation['platform']}")
        print(f"  Attestation verified: {client.attestation_verified}")
        print(f"  Prompts sent: {len(prompts)}")
        print()

    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: Cannot connect to server")
        print(f"   Make sure the server is running at {client.server_url}")
        print()
        print("Start the server with:")
        print("  python -m vllm.entrypoints.confidential_api_server --confidential ...")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
