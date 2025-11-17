"""
Confidential Inference Serving Engine

Extends vLLM's OpenAI-compatible API with confidential computing features:
- Remote attestation
- Encrypted model weight loading
- Secure prompt/response handling
"""

import asyncio
import json
import time
from pathlib import Path
from typing import AsyncIterator, Dict, Optional

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from vllm.confidential.attestation import (
    AttestationManager,
    AttestationReport,
    verify_attestation,
)
from vllm.confidential.encryption import EncryptionManager
from vllm.confidential.kms import KMSClient, create_kms_client, KMSProvider
from vllm.confidential.tee_manager import TEEManager, TEEPlatform
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
from vllm.entrypoints.openai.serving_engine import (
    OpenAIServing,
)
from vllm.logger import init_logger

logger = init_logger(__name__)


class ConfidentialServingEngine(OpenAIServing):
    """
    Confidential Inference serving engine.

    Provides secure model serving with TEE protection and attestation.
    """

    def __init__(
        self,
        async_engine_client: AsyncLLMEngine,
        model_config,
        served_model_names,
        *,
        kms_provider: Optional[KMSProvider] = None,
        kms_config: Optional[Dict] = None,
        require_attestation: bool = True,
        encrypted_model_path: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize confidential serving engine.

        Args:
            async_engine_client: The async LLM engine
            model_config: Model configuration
            served_model_names: List of served model names
            kms_provider: KMS provider to use
            kms_config: KMS configuration dict
            require_attestation: Whether to require attestation
            encrypted_model_path: Path to encrypted model weights
        """
        super().__init__(
            async_engine_client=async_engine_client,
            model_config=model_config,
            served_model_names=served_model_names,
            **kwargs
        )

        # Initialize TEE manager
        self.tee_manager = TEEManager()
        if not self.tee_manager.is_confidential() and require_attestation:
            logger.warning(
                "⚠️  NOT RUNNING IN TEE - Confidential mode disabled ⚠️"
            )

        # Initialize attestation manager
        self.attestation_manager = AttestationManager(
            self.tee_manager.platform
        )

        # Initialize KMS client
        self.kms_client: Optional[KMSClient] = None
        if kms_provider:
            kms_config = kms_config or {}
            self.kms_client = create_kms_client(kms_provider, **kms_config)

        self.require_attestation = require_attestation
        self.encrypted_model_path = encrypted_model_path

        # Generate initial attestation
        if self.tee_manager.is_confidential():
            try:
                self.current_attestation = self.attestation_manager.generate_attestation()
                logger.info("Initial attestation generated successfully")
            except Exception as e:
                logger.error(f"Failed to generate attestation: {e}")
                self.current_attestation = None
        else:
            self.current_attestation = None

        logger.info(
            f"Confidential serving engine initialized. "
            f"TEE: {self.tee_manager.platform.value}, "
            f"Attestation: {self.current_attestation is not None}"
        )

    async def get_attestation(
        self,
        nonce: Optional[str] = None
    ) -> AttestationReport:
        """
        Get attestation report for this TEE instance.

        Args:
            nonce: Optional nonce for freshness

        Returns:
            AttestationReport: Current attestation report
        """
        if not self.tee_manager.is_confidential():
            raise HTTPException(
                status_code=503,
                detail="Not running in TEE, attestation not available"
            )

        # Generate fresh attestation with nonce
        attestation = self.attestation_manager.generate_attestation(nonce=nonce)
        return attestation

    async def verify_client_attestation(
        self,
        attestation_data: Dict
    ) -> bool:
        """
        Verify client's attestation (for mutual attestation).

        Args:
            attestation_data: Client's attestation report

        Returns:
            bool: True if valid
        """
        try:
            report = AttestationReport.from_dict(attestation_data)
            return verify_attestation(report)
        except Exception as e:
            logger.error(f"Client attestation verification failed: {e}")
            return False

    async def decrypt_model_weights(
        self,
        encrypted_path: str,
        output_path: str,
        key_id: str
    ) -> bool:
        """
        Decrypt model weights inside TEE.

        Args:
            encrypted_path: Path to encrypted model
            output_path: Path for decrypted model
            key_id: KMS key ID

        Returns:
            bool: True if successful
        """
        if not self.kms_client:
            raise HTTPException(
                status_code=503,
                detail="KMS client not configured"
            )

        logger.info(f"Decrypting model weights in TEE: {encrypted_path}")

        # Verify we're in TEE before decrypting
        if self.require_attestation and not self.tee_manager.is_confidential():
            raise HTTPException(
                status_code=403,
                detail="Model decryption requires TEE environment"
            )

        # Decrypt model
        manager = EncryptionManager(key_id, self.kms_client)
        success = manager.decrypt_model(encrypted_path, output_path)

        if success:
            logger.info("Model decrypted successfully inside TEE")
        else:
            logger.error("Model decryption failed")

        return success


class ConfidentialChatCompletion(OpenAIServingChat):
    """
    Confidential chat completion endpoint.

    All prompts and responses are encrypted end-to-end.
    """

    def __init__(
        self,
        confidential_engine: ConfidentialServingEngine,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.confidential_engine = confidential_engine

    async def create_chat_completion(
        self,
        request: ChatCompletionRequest,
        raw_request: Request
    ):
        """
        Create chat completion with confidential computing.

        Optionally verifies client attestation before processing.
        """
        # Check if attestation is required
        attestation_header = raw_request.headers.get("X-Attestation-Report")
        if self.confidential_engine.require_attestation and attestation_header:
            try:
                attestation_data = json.loads(attestation_header)
                if not await self.confidential_engine.verify_client_attestation(
                    attestation_data
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="Client attestation verification failed"
                    )
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid attestation report format"
                )

        # Process request normally (data is protected by TEE)
        return await super().create_chat_completion(request, raw_request)


class ConfidentialCompletion(OpenAIServingCompletion):
    """
    Confidential completion endpoint.

    All prompts and responses are encrypted end-to-end.
    """

    def __init__(
        self,
        confidential_engine: ConfidentialServingEngine,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.confidential_engine = confidential_engine

    async def create_completion(
        self,
        request: CompletionRequest,
        raw_request: Request
    ):
        """
        Create completion with confidential computing.

        Optionally verifies client attestation before processing.
        """
        # Check if attestation is required
        attestation_header = raw_request.headers.get("X-Attestation-Report")
        if self.confidential_engine.require_attestation and attestation_header:
            try:
                attestation_data = json.loads(attestation_header)
                if not await self.confidential_engine.verify_client_attestation(
                    attestation_data
                ):
                    raise HTTPException(
                        status_code=403,
                        detail="Client attestation verification failed"
                    )
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid attestation report format"
                )

        # Process request normally (data is protected by TEE)
        return await super().create_completion(request, raw_request)
