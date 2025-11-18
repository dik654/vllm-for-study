use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use shared::{models::HealthResponse, Config};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod handlers;
mod vllm_client;

use handlers::AppState;
use vllm_client::VllmClient;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    // Load configuration
    let config = Config::from_env()?;
    info!("Configuration loaded");

    // Get provider configuration from environment
    let provider_id = std::env::var("PROVIDER_ID").unwrap_or_else(|_| "provider_local".to_string());
    let provider_name = std::env::var("PROVIDER_NAME").unwrap_or_else(|_| "Local Provider".to_string());
    let vllm_host = std::env::var("VLLM_HOST").unwrap_or_else(|_| "localhost".to_string());
    let vllm_port = std::env::var("VLLM_PORT").unwrap_or_else(|_| "8000".to_string());
    let vllm_url = format!("http://{}:{}", vllm_host, vllm_port);

    info!("Provider ID: {}", provider_id);
    info!("Provider Name: {}", provider_name);
    info!("vLLM URL: {}", vllm_url);

    // Create vLLM client
    let vllm_client = VllmClient::new(vllm_url);

    // Create app state
    let state = Arc::new(AppState {
        vllm_client,
        provider_id: provider_id.clone(),
        provider_name: provider_name.clone(),
    });

    // Build our application with routes
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        .route("/vllm/health", get(handlers::vllm_health_check))
        .route("/v1/chat/completions", post(handlers::chat_completions))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Run server
    let port = std::env::var("PROVIDER_PORT").unwrap_or_else(|_| "8001".to_string());
    let addr = format!("{}:{}", config.host, port);
    info!("Provider starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> impl IntoResponse {
    Json(json!({
        "message": "Distributed LLM Platform - Provider Agent",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }))
}

async fn health_check() -> impl IntoResponse {
    let response = HealthResponse {
        status: "healthy".to_string(),
        service: "provider".to_string(),
        version: "0.1.0".to_string(),
    };
    (StatusCode::OK, Json(response))
}
