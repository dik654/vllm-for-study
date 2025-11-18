use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use serde_json::json;
use shared::models::CreateProviderRequest;
use std::sync::Arc;
use uuid::Uuid;

use crate::db::AppState;

// DTOs for User API
#[derive(Debug, Deserialize)]
pub struct CreateUserRequest {
    pub email: String,
    pub tier: Option<String>,
    pub rate_limit_per_minute: Option<i32>,
    pub monthly_budget_usd: Option<Decimal>,
}

#[derive(Debug, Serialize)]
pub struct CreateUserResponse {
    pub user_id: String,
    pub email: String,
    pub api_key: String,
    pub tier: String,
}

// DTOs for Chat Completion
#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<serde_json::Value>,
    pub temperature: Option<f64>,
    pub max_tokens: Option<i32>,
    pub stream: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<serde_json::Value>,
    pub usage: serde_json::Value,
}

// Convert shared::AppError to axum response
fn error_response(err: shared::AppError) -> impl IntoResponse {
    let (status, message) = match err {
        shared::AppError::Database(e) => (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()),
        shared::AppError::Config(e) => (StatusCode::INTERNAL_SERVER_ERROR, e),
        shared::AppError::NotFound(e) => (StatusCode::NOT_FOUND, e),
        shared::AppError::BadRequest(e) => (StatusCode::BAD_REQUEST, e),
        shared::AppError::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized".to_string()),
        shared::AppError::Internal(e) => (StatusCode::INTERNAL_SERVER_ERROR, e),
    };

    (status, Json(json!({ "error": message })))
}

pub async fn create_provider(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateProviderRequest>,
) -> impl IntoResponse {
    let provider_id = format!("provider_{}", Uuid::new_v4());
    let provider_token = format!("prov_tok_{}", Uuid::new_v4());

    let result = sqlx::query_as::<_, shared::models::Provider>(
        r#"
        INSERT INTO providers (
            provider_id, provider_name, endpoint_url, provider_token, status,
            token_input_rate, token_output_rate, cached_input_discount, request_base_rate, batch_discount,
            context_memory_hourly_rate, context_storage_hourly_rate,
            gp_memory_hourly_rate, gp_storage_hot_hourly_rate, gp_storage_cold_hourly_rate,
            max_concurrent_requests, context_memory_capacity_gb, context_storage_capacity_tb,
            gp_memory_capacity_gb, gp_storage_capacity_tb, supported_models
        )
        VALUES ($1, $2, $3, $4, 'active',
                $5, $6, 0.9, $7, 0.5,
                $8, $9,
                $10, $11, $12,
                $13, $14, $15,
                $16, $17, $18)
        RETURNING *
        "#,
    )
    .bind(&provider_id)
    .bind(&req.provider_name)
    .bind(&req.endpoint_url)
    .bind(&provider_token)
    .bind(&req.pricing.token_input_rate)
    .bind(&req.pricing.token_output_rate)
    .bind(rust_decimal::Decimal::new(1, 2))
    .bind(&req.pricing.context_memory_hourly_rate)
    .bind(&req.pricing.context_storage_hourly_rate)
    .bind(&req.pricing.gp_memory_hourly_rate)
    .bind(&req.pricing.gp_storage_hot_hourly_rate)
    .bind(&req.pricing.gp_storage_cold_hourly_rate)
    .bind(req.capacity.max_concurrent_requests)
    .bind(req.capacity.context_memory_capacity_gb)
    .bind(req.capacity.context_storage_capacity_tb)
    .bind(req.capacity.gp_memory_capacity_gb)
    .bind(req.capacity.gp_storage_capacity_tb)
    .bind(sqlx::types::Json(&req.supported_models))
    .fetch_one(&state.pool)
    .await;

    match result {
        Ok(provider) => (StatusCode::CREATED, Json(provider)).into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn list_providers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::Provider>(
        r#"
        SELECT * FROM providers
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .fetch_all(&state.pool)
    .await;

    match result {
        Ok(providers) => (StatusCode::OK, Json(providers)).into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn get_provider(
    State(state): State<Arc<AppState>>,
    Path(provider_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::Provider>(
        r#"
        SELECT * FROM providers
        WHERE provider_id = $1
        "#
    )
    .bind(&provider_id)
    .fetch_optional(&state.pool)
    .await;

    match result {
        Ok(Some(provider)) => (StatusCode::OK, Json(provider)).into_response(),
        Ok(None) => error_response(shared::AppError::NotFound(
            "Provider not found".to_string(),
        ))
        .into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

// User Management Endpoints
pub async fn create_user(
    State(state): State<Arc<AppState>>,
    Json(req): Json<CreateUserRequest>,
) -> impl IntoResponse {
    let user_id = format!("user_{}", Uuid::new_v4());
    let api_key = format!("sk_{}", Uuid::new_v4().simple());
    let tier = req.tier.unwrap_or_else(|| "free".to_string());
    let rate_limit = req.rate_limit_per_minute.unwrap_or(60);

    let result = sqlx::query_as::<_, shared::models::User>(
        r#"
        INSERT INTO users (
            user_id, email, api_key, status, tier,
            billing_email, payment_method_id, credit_balance,
            rate_limit_per_minute, monthly_budget_usd
        )
        VALUES ($1, $2, $3, 'active', $4, NULL, NULL, 0.00, $5, $6)
        RETURNING *
        "#,
    )
    .bind(&user_id)
    .bind(&req.email)
    .bind(&api_key)
    .bind(&tier)
    .bind(rate_limit)
    .bind(req.monthly_budget_usd)
    .fetch_one(&state.pool)
    .await;

    match result {
        Ok(user) => {
            let response = CreateUserResponse {
                user_id: user.user_id,
                email: user.email,
                api_key: user.api_key,
                tier: user.tier,
            };
            (StatusCode::CREATED, Json(response)).into_response()
        }
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn get_user(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::User>(
        r#"
        SELECT * FROM users
        WHERE user_id = $1
        "#
    )
    .bind(&user_id)
    .fetch_optional(&state.pool)
    .await;

    match result {
        Ok(Some(user)) => (StatusCode::OK, Json(user)).into_response(),
        Ok(None) => {
            error_response(shared::AppError::NotFound("User not found".to_string()))
                .into_response()
        }
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn list_users(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::User>(
        r#"
        SELECT * FROM users
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .fetch_all(&state.pool)
    .await;

    match result {
        Ok(users) => (StatusCode::OK, Json(users)).into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

// Chat Completion Proxy (placeholder - will be implemented with provider integration)
pub async fn chat_completions(
    State(_state): State<Arc<AppState>>,
    Json(req): Json<ChatCompletionRequest>,
) -> impl IntoResponse {
    // TODO: Implement actual provider routing and request forwarding
    // For now, return a placeholder response
    let response = ChatCompletionResponse {
        id: format!("chatcmpl_{}", Uuid::new_v4()),
        object: "chat.completion".to_string(),
        created: chrono::Utc::now().timestamp(),
        model: req.model.clone(),
        choices: vec![json!({
            "index": 0,
            "message": {
                "role": "assistant",
                "content": "This is a placeholder response. Provider integration pending."
            },
            "finish_reason": "stop"
        })],
        usage: json!({
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20
        }),
    };

    (StatusCode::OK, Json(response)).into_response()
}

// Request logging
pub async fn list_requests(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::Request>(
        r#"
        SELECT * FROM requests
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .bind(&user_id)
    .fetch_all(&state.pool)
    .await;

    match result {
        Ok(requests) => (StatusCode::OK, Json(requests)).into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

// Workflow Management
#[derive(Debug, Deserialize)]
pub struct CreateWorkflowRequest {
    pub name: String,
    pub description: Option<String>,
    pub nodes: serde_json::Value,
    pub connections: serde_json::Value,
    pub metadata: Option<serde_json::Value>,
    pub settings: Option<serde_json::Value>,
}

pub async fn create_workflow(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
    Json(req): Json<CreateWorkflowRequest>,
) -> impl IntoResponse {
    let workflow_id = format!("workflow_{}", Uuid::new_v4());
    let metadata = req.metadata.unwrap_or_else(|| json!({}));
    let settings = req.settings.unwrap_or_else(|| json!({}));

    let result = sqlx::query_as::<_, shared::models::Workflow>(
        r#"
        INSERT INTO workflows (
            workflow_id, user_id, name, description, version,
            nodes, connections, metadata, settings,
            is_public, fork_count, execution_count
        )
        VALUES ($1, $2, $3, $4, '1.0.0', $5, $6, $7, $8, false, 0, 0)
        RETURNING *
        "#,
    )
    .bind(&workflow_id)
    .bind(&user_id)
    .bind(&req.name)
    .bind(&req.description)
    .bind(sqlx::types::Json(&req.nodes))
    .bind(sqlx::types::Json(&req.connections))
    .bind(sqlx::types::Json(&metadata))
    .bind(sqlx::types::Json(&settings))
    .fetch_one(&state.pool)
    .await;

    match result {
        Ok(workflow) => (StatusCode::CREATED, Json(workflow)).into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn get_workflow(
    State(state): State<Arc<AppState>>,
    Path(workflow_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::Workflow>(
        r#"
        SELECT * FROM workflows
        WHERE workflow_id = $1
        "#
    )
    .bind(&workflow_id)
    .fetch_optional(&state.pool)
    .await;

    match result {
        Ok(Some(workflow)) => (StatusCode::OK, Json(workflow)).into_response(),
        Ok(None) => error_response(shared::AppError::NotFound(
            "Workflow not found".to_string(),
        ))
        .into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn list_workflows(
    State(state): State<Arc<AppState>>,
    Path(user_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::Workflow>(
        r#"
        SELECT * FROM workflows
        WHERE user_id = $1
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .bind(&user_id)
    .fetch_all(&state.pool)
    .await;

    match result {
        Ok(workflows) => (StatusCode::OK, Json(workflows)).into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn update_workflow(
    State(state): State<Arc<AppState>>,
    Path(workflow_id): Path<String>,
    Json(req): Json<CreateWorkflowRequest>,
) -> impl IntoResponse {
    let metadata = req.metadata.unwrap_or_else(|| json!({}));
    let settings = req.settings.unwrap_or_else(|| json!({}));

    let result = sqlx::query_as::<_, shared::models::Workflow>(
        r#"
        UPDATE workflows
        SET name = $1, description = $2, nodes = $3, connections = $4,
            metadata = $5, settings = $6, updated_at = NOW()
        WHERE workflow_id = $7
        RETURNING *
        "#,
    )
    .bind(&req.name)
    .bind(&req.description)
    .bind(sqlx::types::Json(&req.nodes))
    .bind(sqlx::types::Json(&req.connections))
    .bind(sqlx::types::Json(&metadata))
    .bind(sqlx::types::Json(&settings))
    .bind(&workflow_id)
    .fetch_optional(&state.pool)
    .await;

    match result {
        Ok(Some(workflow)) => (StatusCode::OK, Json(workflow)).into_response(),
        Ok(None) => error_response(shared::AppError::NotFound(
            "Workflow not found".to_string(),
        ))
        .into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn delete_workflow(
    State(state): State<Arc<AppState>>,
    Path(workflow_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query(
        r#"
        DELETE FROM workflows
        WHERE workflow_id = $1
        "#
    )
    .bind(&workflow_id)
    .execute(&state.pool)
    .await;

    match result {
        Ok(result) if result.rows_affected() > 0 => {
            (StatusCode::NO_CONTENT, Json(json!({}))).into_response()
        }
        Ok(_) => error_response(shared::AppError::NotFound(
            "Workflow not found".to_string(),
        ))
        .into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

// Workflow Execution
#[derive(Debug, Deserialize)]
pub struct ExecuteWorkflowRequest {
    pub input_data: serde_json::Value,
}

pub async fn execute_workflow(
    State(state): State<Arc<AppState>>,
    Path((workflow_id, user_id)): Path<(String, String)>,
    Json(req): Json<ExecuteWorkflowRequest>,
) -> impl IntoResponse {
    let execution_id = format!("exec_{}", Uuid::new_v4());

    let result = sqlx::query_as::<_, shared::models::WorkflowExecution>(
        r#"
        INSERT INTO workflow_executions (
            execution_id, workflow_id, user_id, state, input_data,
            output_data, progress, node_outputs, started_at, completed_at,
            total_duration_ms, total_cost_usd, error_message
        )
        VALUES ($1, $2, $3, 'pending', $4, NULL, NULL, NULL, NOW(), NULL, NULL, NULL, NULL)
        RETURNING *
        "#,
    )
    .bind(&execution_id)
    .bind(&workflow_id)
    .bind(&user_id)
    .bind(sqlx::types::Json(&req.input_data))
    .fetch_one(&state.pool)
    .await;

    match result {
        Ok(execution) => {
            // TODO: Trigger async workflow execution
            (StatusCode::CREATED, Json(execution)).into_response()
        }
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}

pub async fn get_execution(
    State(state): State<Arc<AppState>>,
    Path(execution_id): Path<String>,
) -> impl IntoResponse {
    let result = sqlx::query_as::<_, shared::models::WorkflowExecution>(
        r#"
        SELECT * FROM workflow_executions
        WHERE execution_id = $1
        "#
    )
    .bind(&execution_id)
    .fetch_optional(&state.pool)
    .await;

    match result {
        Ok(Some(execution)) => (StatusCode::OK, Json(execution)).into_response(),
        Ok(None) => error_response(shared::AppError::NotFound(
            "Execution not found".to_string(),
        ))
        .into_response(),
        Err(e) => error_response(shared::AppError::Database(e)).into_response(),
    }
}
