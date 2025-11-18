"""
Run vLLM confidential computing CLI

Allows running: python -m vllm.confidential
"""

from vllm.entrypoints.cli.confidential_cli import main

if __name__ == "__main__":
    main()
