use axum::{
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use serde_json::json;
use shared::{models::HealthResponse, Config};
use sqlx::postgres::PgPoolOptions;
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod db;
mod handlers;

use db::AppState;

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

    // Create database pool
    let pool = PgPoolOptions::new()
        .max_connections(20)
        .connect(&config.database_url)
        .await?;
    info!("Database connection established");

    // Run migrations (if needed)
    // sqlx::migrate!("./migrations").run(&pool).await?;

    // Create app state
    let state = Arc::new(AppState { pool });

    // Build our application with routes
    let app = Router::new()
        .route("/", get(root))
        .route("/health", get(health_check))
        // Provider routes
        .route("/v1/providers", post(handlers::create_provider))
        .route("/v1/providers", get(handlers::list_providers))
        .route("/v1/providers/:provider_id", get(handlers::get_provider))
        // User routes
        .route("/v1/users", post(handlers::create_user))
        .route("/v1/users", get(handlers::list_users))
        .route("/v1/users/:user_id", get(handlers::get_user))
        .route("/v1/users/:user_id/requests", get(handlers::list_requests))
        // LLM API routes
        .route("/v1/chat/completions", post(handlers::chat_completions))
        // Workflow routes
        .route("/v1/users/:user_id/workflows", post(handlers::create_workflow))
        .route("/v1/users/:user_id/workflows", get(handlers::list_workflows))
        .route("/v1/workflows/:workflow_id", get(handlers::get_workflow))
        .route("/v1/workflows/:workflow_id", axum::routing::put(handlers::update_workflow))
        .route("/v1/workflows/:workflow_id", axum::routing::delete(handlers::delete_workflow))
        // Workflow execution routes
        .route("/v1/workflows/:workflow_id/execute/:user_id", post(handlers::execute_workflow))
        .route("/v1/executions/:execution_id", get(handlers::get_execution))
        .with_state(state)
        .layer(
            CorsLayer::new()
                .allow_origin(Any)
                .allow_methods(Any)
                .allow_headers(Any),
        );

    // Run server
    let addr = format!("{}:{}", config.host, config.port);
    info!("Router starting on {}", addr);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn root() -> impl IntoResponse {
    Json(json!({
        "message": "Distributed LLM Platform - Central Router",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health"
    }))
}

async fn health_check() -> impl IntoResponse {
    let response = HealthResponse {
        status: "healthy".to_string(),
        service: "router".to_string(),
        version: "0.1.0".to_string(),
    };
    (StatusCode::OK, Json(response))
}
