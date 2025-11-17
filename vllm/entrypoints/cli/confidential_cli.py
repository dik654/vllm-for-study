#!/usr/bin/env python3
"""
vLLM Confidential Computing CLI

Command-line tool for managing confidential AI inference:
- Encrypt model weights
- Generate attestation reports
- Verify attestation
- Manage KMS keys
"""

import argparse
import json
import sys
from pathlib import Path

from vllm.confidential.attestation import AttestationManager, verify_attestation, AttestationReport
from vllm.confidential.encryption import encrypt_model_weights, decrypt_model_weights
from vllm.confidential.kms import create_kms_client, KMSProvider
from vllm.confidential.tee_manager import TEEManager
from vllm.logger import init_logger

logger = init_logger(__name__)


def cmd_status(args):
    """Show confidential computing status"""
    tee_manager = TEEManager()
    info = tee_manager.get_platform_info()

    print("╔══════════════════════════════════════════════════════╗")
    print("║     vLLM Confidential Computing Status              ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()
    print(f"Platform:             {info['platform']}")
    print(f"Confidential:         {'✓ YES' if info['confidential'] else '✗ NO'}")
    print()
    print("Features:")
    for feature, enabled in info['features'].items():
        status = "✓" if enabled else "✗"
        print(f"  {status} {feature.replace('_', ' ').title()}")
    print()

    if not info['confidential']:
        print("⚠️  WARNING: Not running in a TEE environment!")
        print("   Confidential computing features are NOT available.")
        print("   Model weights and prompts will NOT be protected.")
        print()


def cmd_attestation(args):
    """Generate attestation report"""
    tee_manager = TEEManager()

    if not tee_manager.is_confidential():
        print("⚠️  ERROR: Not running in a TEE environment!")
        print("   Cannot generate genuine attestation.")
        sys.exit(1)

    print("Generating attestation report...")
    attestation_manager = AttestationManager(tee_manager.platform)

    nonce = args.nonce if hasattr(args, 'nonce') else None
    attestation = attestation_manager.generate_attestation(nonce=nonce)

    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(attestation.to_dict(), f, indent=2)
        print(f"✓ Attestation saved to: {output_path}")
    else:
        print(json.dumps(attestation.to_dict(), indent=2))


def cmd_verify_attestation(args):
    """Verify attestation report"""
    attestation_path = Path(args.attestation_file)

    if not attestation_path.exists():
        print(f"⚠️  ERROR: Attestation file not found: {attestation_path}")
        sys.exit(1)

    with open(attestation_path, 'r') as f:
        attestation_data = json.load(f)

    attestation = AttestationReport.from_dict(attestation_data)

    print(f"Verifying attestation from: {attestation_path}")
    print(f"Platform: {attestation.platform.value}")
    print(f"Measurement: {attestation.measurement}")
    print(f"Timestamp: {attestation.timestamp}")
    print()

    try:
        expected_measurement = args.expected_measurement if hasattr(args, 'expected_measurement') else None
        valid = verify_attestation(
            attestation,
            expected_measurement=expected_measurement
        )

        if valid:
            print("✓ Attestation is VALID")
            sys.exit(0)
        else:
            print("✗ Attestation is INVALID")
            sys.exit(1)
    except Exception as e:
        print(f"✗ Attestation verification FAILED: {e}")
        sys.exit(1)


def cmd_encrypt_model(args):
    """Encrypt model weights"""
    model_path = Path(args.model_path)
    output_path = Path(args.output_path)
    key_id = args.key_id

    if not model_path.exists():
        print(f"⚠️  ERROR: Model path not found: {model_path}")
        sys.exit(1)

    print(f"Encrypting model: {model_path}")
    print(f"Output path: {output_path}")
    print(f"KMS Key ID: {key_id}")
    print()

    # Initialize KMS client
    kms_client = None
    if args.kms_provider:
        kms_config = json.loads(args.kms_config) if args.kms_config else {}
        kms_client = create_kms_client(
            KMSProvider(args.kms_provider),
            **kms_config
        )
        print(f"Using KMS provider: {args.kms_provider}")
    else:
        print("⚠️  WARNING: No KMS provider specified, using local encryption")
        print("   This is NOT secure for production use!")

    print()
    print("Encrypting model files...")

    try:
        metadata = encrypt_model_weights(
            model_path=model_path,
            output_path=output_path,
            key_id=key_id,
            kms_client=kms_client
        )

        print()
        print("✓ Model encrypted successfully!")
        print(f"  Encrypted files: {len(metadata.encrypted_files)}")
        print(f"  Algorithm: {metadata.algorithm}")
        print()
        print("To load this model in vLLM:")
        print(f"  python -m vllm.entrypoints.confidential_api_server \\")
        print(f"    --confidential \\")
        print(f"    --encrypted-model-path {output_path} \\")
        print(f"    --model-encryption-key-id {key_id} \\")
        if args.kms_provider:
            print(f"    --kms-provider {args.kms_provider}")
        print()

    except Exception as e:
        print(f"✗ Encryption FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def cmd_decrypt_model(args):
    """Decrypt model weights"""
    encrypted_path = Path(args.encrypted_path)
    output_path = Path(args.output_path)
    key_id = args.key_id

    if not encrypted_path.exists():
        print(f"⚠️  ERROR: Encrypted model path not found: {encrypted_path}")
        sys.exit(1)

    print(f"Decrypting model: {encrypted_path}")
    print(f"Output path: {output_path}")
    print(f"KMS Key ID: {key_id}")
    print()

    # Initialize KMS client
    kms_client = None
    if args.kms_provider:
        kms_config = json.loads(args.kms_config) if args.kms_config else {}
        kms_client = create_kms_client(
            KMSProvider(args.kms_provider),
            **kms_config
        )
        print(f"Using KMS provider: {args.kms_provider}")
    else:
        print("⚠️  WARNING: No KMS provider specified, using local decryption")

    # Check if in TEE
    tee_manager = TEEManager()
    if not tee_manager.is_confidential():
        print()
        print("⚠️  WARNING: Not running in TEE!")
        print("   Decrypted model will NOT be protected.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Aborted.")
            sys.exit(0)

    print()
    print("Decrypting model files...")

    try:
        success = decrypt_model_weights(
            encrypted_path=encrypted_path,
            output_path=output_path,
            key_id=key_id,
            kms_client=kms_client
        )

        if success:
            print()
            print("✓ Model decrypted successfully!")
            print(f"  Decrypted model: {output_path}")
        else:
            print()
            print("✗ Decryption FAILED")
            sys.exit(1)

    except Exception as e:
        print(f"✗ Decryption FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="vLLM Confidential Computing CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Status command
    status_parser = subparsers.add_parser(
        'status',
        help='Show confidential computing status'
    )
    status_parser.set_defaults(func=cmd_status)

    # Attestation command
    attest_parser = subparsers.add_parser(
        'attestation',
        help='Generate attestation report'
    )
    attest_parser.add_argument(
        '--nonce',
        type=str,
        help='Nonce for freshness proof'
    )
    attest_parser.add_argument(
        '--output',
        '-o',
        type=str,
        help='Output file for attestation report'
    )
    attest_parser.set_defaults(func=cmd_attestation)

    # Verify attestation command
    verify_parser = subparsers.add_parser(
        'verify',
        help='Verify attestation report'
    )
    verify_parser.add_argument(
        'attestation_file',
        type=str,
        help='Path to attestation report JSON'
    )
    verify_parser.add_argument(
        '--expected-measurement',
        type=str,
        help='Expected measurement hash'
    )
    verify_parser.set_defaults(func=cmd_verify_attestation)

    # Encrypt model command
    encrypt_parser = subparsers.add_parser(
        'encrypt',
        help='Encrypt model weights'
    )
    encrypt_parser.add_argument(
        'model_path',
        type=str,
        help='Path to unencrypted model directory'
    )
    encrypt_parser.add_argument(
        'output_path',
        type=str,
        help='Output path for encrypted model'
    )
    encrypt_parser.add_argument(
        '--key-id',
        type=str,
        required=True,
        help='KMS key ID for encryption'
    )
    encrypt_parser.add_argument(
        '--kms-provider',
        type=str,
        choices=[p.value for p in KMSProvider],
        help='KMS provider (default: local)'
    )
    encrypt_parser.add_argument(
        '--kms-config',
        type=str,
        help='KMS configuration JSON'
    )
    encrypt_parser.set_defaults(func=cmd_encrypt_model)

    # Decrypt model command
    decrypt_parser = subparsers.add_parser(
        'decrypt',
        help='Decrypt model weights'
    )
    decrypt_parser.add_argument(
        'encrypted_path',
        type=str,
        help='Path to encrypted model directory'
    )
    decrypt_parser.add_argument(
        'output_path',
        type=str,
        help='Output path for decrypted model'
    )
    decrypt_parser.add_argument(
        '--key-id',
        type=str,
        required=True,
        help='KMS key ID for decryption'
    )
    decrypt_parser.add_argument(
        '--kms-provider',
        type=str,
        choices=[p.value for p in KMSProvider],
        help='KMS provider (default: local)'
    )
    decrypt_parser.add_argument(
        '--kms-config',
        type=str,
        help='KMS configuration JSON'
    )
    decrypt_parser.set_defaults(func=cmd_decrypt_model)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
