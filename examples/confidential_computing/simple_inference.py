#!/usr/bin/env python3
"""
Simple Confidential Inference Example

Demonstrates basic usage of vLLM confidential computing:
1. Check TEE status
2. Get server attestation
3. Run inference
"""

import json
import requests

# Server endpoint
SERVER_URL = "http://localhost:8000"


def check_tee_status():
    """Check if server is running in TEE"""
    print("=" * 60)
    print("Checking Confidential Computing Status")
    print("=" * 60)

    response = requests.get(f"{SERVER_URL}/v1/confidential/status")
    status = response.json()

    print(f"Platform: {status['platform']}")
    print(f"Confidential: {'✓ YES' if status['confidential'] else '✗ NO'}")
    print(f"Attestation Available: {'✓ YES' if status['attestation_available'] else '✗ NO'}")
    print(f"GPU TEE: {'✓ YES' if status['gpu_tee'] else '✗ NO'}")
    print()

    return status['confidential']


def get_attestation():
    """Get attestation report from server"""
    print("=" * 60)
    print("Requesting Attestation Report")
    print("=" * 60)

    # Generate random nonce for freshness
    import hashlib
    import time
    nonce = hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]

    response = requests.post(
        f"{SERVER_URL}/v1/confidential/attestation",
        json={"nonce": nonce}
    )

    if response.status_code != 200:
        print(f"⚠️  Attestation request failed: {response.text}")
        return None

    attestation = response.json()
    print(f"Platform: {attestation['platform']}")
    print(f"Measurement: {attestation['measurement'][:32]}...")
    print(f"Timestamp: {attestation['timestamp']}")
    print(f"Nonce: {attestation['nonce']}")
    print()

    # In production, you would:
    # 1. Verify the attestation signature
    # 2. Check the measurement matches expected value
    # 3. Verify certificate chain
    # 4. Check timestamp freshness

    return attestation


def run_inference(prompt: str):
    """Run inference on the confidential server"""
    print("=" * 60)
    print("Running Confidential Inference")
    print("=" * 60)
    print(f"Prompt: {prompt}")
    print()

    response = requests.post(
        f"{SERVER_URL}/v1/chat/completions",
        json={
            "model": "your-model",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100
        }
    )

    if response.status_code != 200:
        print(f"⚠️  Inference request failed: {response.text}")
        return None

    result = response.json()
    assistant_message = result['choices'][0]['message']['content']

    print(f"Response: {assistant_message}")
    print()

    return result


def main():
    """Main example workflow"""
    print("\n" + "=" * 60)
    print("vLLM Confidential Inference Example")
    print("=" * 60 + "\n")

    try:
        # Step 1: Check TEE status
        is_confidential = check_tee_status()

        if not is_confidential:
            print("⚠️  WARNING: Server is NOT running in TEE!")
            print("   Prompts and model weights are NOT protected.")
            response = input("\nContinue anyway? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return

        # Step 2: Get and verify attestation
        attestation = get_attestation()

        if attestation:
            print("✓ Attestation received successfully")
            print("  In production, you should verify:")
            print("  - Signature and certificate chain")
            print("  - Measurement matches expected value")
            print("  - Timestamp is recent")
            print()

        # Step 3: Run inference
        prompts = [
            "What is confidential computing?",
            "Explain TEE in simple terms.",
            "Why is model privacy important?"
        ]

        for prompt in prompts:
            result = run_inference(prompt)
            if result:
                print("✓ Inference completed successfully")
                print()

        print("=" * 60)
        print("Example completed successfully!")
        print("=" * 60)

    except requests.exceptions.ConnectionError:
        print("⚠️  ERROR: Cannot connect to server")
        print(f"   Make sure the server is running at {SERVER_URL}")
        print()
        print("Start the server with:")
        print("  python -m vllm.entrypoints.confidential_api_server --confidential ...")
    except Exception as e:
        print(f"⚠️  ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
