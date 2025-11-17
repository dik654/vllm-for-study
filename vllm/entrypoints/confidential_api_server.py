"""
Confidential AI Inference API Server

FastAPI server for confidential LLM inference with TEE protection.

Features:
- Remote attestation endpoints
- Encrypted model weight upload and loading
- Secure inference with end-to-end encryption
- KMS integration for key management
"""

import argparse
import asyncio
import json
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from fastapi.middleware.cors import CORSMiddleware

from vllm.confidential.attestation import AttestationManager
from vllm.confidential.encryption import EncryptionManager
from vllm.confidential.kms import KMSProvider, create_kms_client
from vllm.confidential.tee_manager import TEEManager
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.api_server import (
    build_async_engine_client,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    CompletionRequest,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_confidential import (
    ConfidentialChatCompletion,
    ConfidentialCompletion,
    ConfidentialServingEngine,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Global state
engine_client: Optional[AsyncLLMEngine] = None
confidential_engine: Optional[ConfidentialServingEngine] = None
tee_manager: Optional[TEEManager] = None
attestation_manager: Optional[AttestationManager] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup/shutdown"""
    # Startup
    logger.info("Starting Confidential AI Inference API Server")

    # Initialize global TEE manager
    global tee_manager, attestation_manager
    tee_manager = TEEManager()
    attestation_manager = AttestationManager(tee_manager.platform)

    if tee_manager.is_confidential():
        logger.info(f"✓ Running in TEE: {tee_manager.platform.value}")
    else:
        logger.warning("⚠️  NOT running in TEE - confidentiality NOT guaranteed")

    yield

    # Shutdown
    logger.info("Shutting down Confidential AI Inference API Server")


# Create FastAPI app
app = FastAPI(
    title="vLLM Confidential Inference API",
    description="Secure LLM inference with Trusted Execution Environments",
    version="0.1.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Confidential Computing Endpoints ====================

@app.get("/v1/confidential/status")
async def get_confidential_status():
    """
    Get confidential computing status.

    Returns information about TEE platform and security features.
    """
    if not tee_manager:
        raise HTTPException(status_code=503, detail="TEE manager not initialized")

    return JSONResponse(content={
        "confidential": tee_manager.is_confidential(),
        "platform": tee_manager.platform.value,
        "features": tee_manager.features,
        "attestation_available": tee_manager.supports_attestation(),
        "gpu_tee": tee_manager.supports_gpu_tee(),
    })


@app.post("/v1/confidential/attestation")
async def get_attestation(request: Request):
    """
    Generate remote attestation report.

    Request body:
        {
            "nonce": "optional_nonce_for_freshness"
        }

    Returns attestation report proving code is running in genuine TEE.
    """
    if not tee_manager or not attestation_manager:
        raise HTTPException(status_code=503, detail="Attestation not available")

    if not tee_manager.supports_attestation():
        raise HTTPException(
            status_code=503,
            detail=f"Attestation not supported on platform: {tee_manager.platform}"
        )

    try:
        body = await request.json() if request.headers.get("content-type") else {}
        nonce = body.get("nonce")

        attestation = attestation_manager.generate_attestation(nonce=nonce)

        return JSONResponse(content=attestation.to_dict())
    except Exception as e:
        logger.error(f"Attestation generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate attestation: {str(e)}"
        )


@app.post("/v1/confidential/verify-attestation")
async def verify_attestation_endpoint(request: Request):
    """
    Verify a remote attestation report.

    Request body:
        {
            "attestation": { ... attestation report ... },
            "expected_measurement": "optional_expected_hash"
        }
    """
    try:
        from vllm.confidential.attestation import (
            AttestationReport,
            verify_attestation,
        )

        body = await request.json()
        attestation_data = body.get("attestation")
        expected_measurement = body.get("expected_measurement")

        if not attestation_data:
            raise HTTPException(
                status_code=400,
                detail="Missing 'attestation' in request body"
            )

        attestation = AttestationReport.from_dict(attestation_data)
        valid = verify_attestation(
            attestation,
            expected_measurement=expected_measurement
        )

        return JSONResponse(content={
            "valid": valid,
            "platform": attestation.platform.value,
            "measurement": attestation.measurement,
            "timestamp": attestation.timestamp.isoformat(),
        })
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Attestation verification failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Verification failed: {str(e)}"
        )


@app.post("/v1/confidential/models/upload")
async def upload_encrypted_model(request: Request):
    """
    Upload encrypted model weights.

    This endpoint accepts encrypted model weights and metadata.
    The weights will be decrypted inside the TEE using the KMS.

    Request should be multipart/form-data with:
    - model_file: encrypted model archive
    - metadata: encryption metadata JSON
    - key_id: KMS key ID
    """
    raise HTTPException(
        status_code=501,
        detail="Model upload endpoint not yet implemented. "
               "Use CLI tool to encrypt and load models."
    )


@app.post("/v1/confidential/models/decrypt")
async def decrypt_model_weights(request: Request):
    """
    Decrypt model weights inside TEE.

    Request body:
        {
            "encrypted_path": "/path/to/encrypted/model",
            "output_path": "/path/to/decrypted/model",
            "key_id": "kms_key_id"
        }

    This endpoint is typically called during initialization to load
    encrypted models into the TEE.
    """
    if not confidential_engine:
        raise HTTPException(
            status_code=503,
            detail="Confidential engine not initialized"
        )

    try:
        body = await request.json()
        encrypted_path = body.get("encrypted_path")
        output_path = body.get("output_path")
        key_id = body.get("key_id")

        if not all([encrypted_path, output_path, key_id]):
            raise HTTPException(
                status_code=400,
                detail="Missing required fields: encrypted_path, output_path, key_id"
            )

        success = await confidential_engine.decrypt_model_weights(
            encrypted_path, output_path, key_id
        )

        return JSONResponse(content={
            "success": success,
            "decrypted_path": output_path if success else None
        })
    except Exception as e:
        logger.error(f"Model decryption failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Decryption failed: {str(e)}"
        )


# ==================== Confidential Inference Endpoints ====================

@app.post("/v1/confidential/chat/completions")
async def confidential_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    """
    Create chat completion with confidential computing.

    Same as /v1/chat/completions but with optional attestation verification.

    Add X-Attestation-Report header for mutual attestation.
    """
    if not confidential_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Delegate to confidential chat handler
    # TODO: Implement proper serving_confidential integration
    raise HTTPException(
        status_code=501,
        detail="Confidential chat endpoint not yet fully implemented"
    )


@app.post("/v1/confidential/completions")
async def confidential_completion(request: CompletionRequest, raw_request: Request):
    """
    Create completion with confidential computing.

    Same as /v1/completions but with optional attestation verification.

    Add X-Attestation-Report header for mutual attestation.
    """
    if not confidential_engine:
        raise HTTPException(status_code=503, detail="Engine not initialized")

    # Delegate to confidential completion handler
    # TODO: Implement proper serving_confidential integration
    raise HTTPException(
        status_code=501,
        detail="Confidential completion endpoint not yet fully implemented"
    )


# ==================== Health Check ====================

@app.get("/health")
async def health():
    """Health check endpoint"""
    return JSONResponse(content={
        "status": "healthy",
        "confidential": tee_manager.is_confidential() if tee_manager else False
    })


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return JSONResponse(content={
        "name": "vLLM Confidential Inference API",
        "version": "0.1.0",
        "confidential": tee_manager.is_confidential() if tee_manager else False,
        "platform": tee_manager.platform.value if tee_manager else "unknown",
        "endpoints": {
            "status": "/v1/confidential/status",
            "attestation": "/v1/confidential/attestation",
            "verify": "/v1/confidential/verify-attestation",
            "health": "/health",
        }
    })


def parse_args():
    """Parse command-line arguments"""
    parser = make_arg_parser()

    # Add confidential computing arguments
    parser.add_argument(
        "--confidential",
        action="store_true",
        help="Enable confidential computing mode"
    )
    parser.add_argument(
        "--require-attestation",
        action="store_true",
        help="Require client attestation for inference requests"
    )
    parser.add_argument(
        "--kms-provider",
        type=str,
        choices=[p.value for p in KMSProvider],
        help="KMS provider for key management"
    )
    parser.add_argument(
        "--kms-config",
        type=str,
        help="KMS configuration JSON"
    )
    parser.add_argument(
        "--encrypted-model-path",
        type=str,
        help="Path to encrypted model weights"
    )
    parser.add_argument(
        "--model-encryption-key-id",
        type=str,
        help="KMS key ID for model decryption"
    )

    return parser.parse_args()


async def run_server(args):
    """Run the confidential inference server"""
    import uvicorn

    # Build async engine
    global engine_client, confidential_engine
    engine_args = AsyncEngineArgs.from_cli_args(args)
    engine_client = await build_async_engine_client(engine_args)

    # Initialize confidential engine if enabled
    if args.confidential:
        kms_config = json.loads(args.kms_config) if args.kms_config else {}

        confidential_engine = ConfidentialServingEngine(
            async_engine_client=engine_client,
            model_config=engine_client.engine.get_model_config(),
            served_model_names=[args.model] if hasattr(args, 'model') else [],
            kms_provider=KMSProvider(args.kms_provider) if args.kms_provider else None,
            kms_config=kms_config,
            require_attestation=args.require_attestation,
            encrypted_model_path=args.encrypted_model_path,
        )

        logger.info("✓ Confidential engine initialized")

    # Run server
    config = uvicorn.Config(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=5,
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_server(args))
