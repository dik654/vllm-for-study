use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde_json::json;
use std::sync::Arc;
use tracing::{error, info};

use crate::vllm_client::{ChatCompletionRequest, VllmClient};

#[derive(Clone)]
pub struct AppState {
    pub vllm_client: VllmClient,
    pub provider_id: String,
    pub provider_name: String,
}

pub async fn chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    info!(
        "Received chat completion request for model: {} from provider: {}",
        req.model, state.provider_name
    );

    match state.vllm_client.chat_completions(req).await {
        Ok(response) => {
            info!("Chat completion successful");
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(e) => {
            error!("Chat completion error: {}", e);
            let error_response = json!({
                "error": {
                    "message": format!("vLLM API error: {}", e),
                    "type": "provider_error",
                    "code": "vllm_error"
                }
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(error_response)).into_response()
        }
    }
}

pub async fn vllm_health_check(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    match state.vllm_client.health_check().await {
        Ok(true) => {
            let response = json!({
                "status": "healthy",
                "provider_id": state.provider_id,
                "provider_name": state.provider_name,
                "vllm_status": "ready"
            });
            (StatusCode::OK, Json(response)).into_response()
        }
        Ok(false) | Err(_) => {
            let response = json!({
                "status": "unhealthy",
                "provider_id": state.provider_id,
                "provider_name": state.provider_name,
                "vllm_status": "not_ready"
            });
            (StatusCode::SERVICE_UNAVAILABLE, Json(response)).into_response()
        }
    }
}
