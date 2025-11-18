use chrono::{DateTime, Utc};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use sqlx::FromRow;

// Provider model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Provider {
    pub provider_id: String,
    pub provider_name: String,
    pub endpoint_url: String,
    pub provider_token: String,
    pub status: String,

    // LLM Pricing
    pub token_input_rate: Decimal,
    pub token_output_rate: Decimal,
    pub cached_input_discount: Decimal,
    pub request_base_rate: Decimal,
    pub batch_discount: Decimal,

    // Context Memory/Storage Pricing
    pub context_memory_hourly_rate: Decimal,
    pub context_storage_hourly_rate: Decimal,

    // GP Memory/Storage Pricing
    pub gp_memory_hourly_rate: Decimal,
    pub gp_storage_hot_hourly_rate: Decimal,
    pub gp_storage_cold_hourly_rate: Decimal,

    // Capacity
    pub max_concurrent_requests: i32,
    pub context_memory_capacity_gb: i32,
    pub context_storage_capacity_tb: i32,
    pub gp_memory_capacity_gb: i32,
    pub gp_storage_capacity_tb: i32,

    // Metadata
    pub supported_models: sqlx::types::Json<Vec<String>>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// User model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct User {
    pub user_id: String,
    pub email: String,
    pub api_key: String,
    pub status: String,
    pub tier: String,

    // Billing
    pub billing_email: Option<String>,
    pub payment_method_id: Option<String>,
    pub credit_balance: Decimal,

    // Limits
    pub rate_limit_per_minute: i32,
    pub monthly_budget_usd: Option<Decimal>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Request log model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Request {
    pub request_id: String,
    pub user_id: String,
    pub provider_id: String,

    // Request Info
    pub model: String,
    pub endpoint: String,
    pub is_batch: bool,

    // Token Usage
    pub input_tokens: i32,
    pub cached_input_tokens: i32,
    pub output_tokens: i32,
    pub latency_ms: i32,

    // Cost Breakdown
    pub provider_cost_usd: Decimal,
    pub commission_usd: Decimal,
    pub user_charge_usd: Decimal,

    // Status
    pub status: String,
    pub error_message: Option<String>,

    pub created_at: DateTime<Utc>,
}

// Workflow model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct Workflow {
    pub workflow_id: String,
    pub user_id: String,
    pub name: String,
    pub description: Option<String>,
    pub version: String,

    // Workflow Definition (JSON)
    pub nodes: sqlx::types::Json<serde_json::Value>,
    pub connections: sqlx::types::Json<serde_json::Value>,
    pub metadata: sqlx::types::Json<serde_json::Value>,
    pub settings: sqlx::types::Json<serde_json::Value>,

    pub is_public: bool,
    pub fork_count: i32,
    pub execution_count: i32,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

// Workflow Execution model
#[derive(Debug, Clone, Serialize, Deserialize, FromRow)]
pub struct WorkflowExecution {
    pub execution_id: String,
    pub workflow_id: String,
    pub user_id: String,

    pub state: String,
    pub input_data: sqlx::types::Json<serde_json::Value>,
    pub output_data: Option<sqlx::types::Json<serde_json::Value>>,

    pub progress: Option<sqlx::types::Json<serde_json::Value>>,
    pub node_outputs: Option<sqlx::types::Json<serde_json::Value>>,

    pub started_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,

    pub total_duration_ms: Option<i32>,
    pub total_cost_usd: Option<Decimal>,

    pub error_message: Option<String>,
}

// DTOs
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
    pub version: String,
}

#[derive(Debug, Deserialize)]
pub struct CreateProviderRequest {
    pub provider_name: String,
    pub endpoint_url: String,
    pub supported_models: Vec<String>,
    pub pricing: PricingConfig,
    pub capacity: CapacityConfig,
}

#[derive(Debug, Deserialize)]
pub struct PricingConfig {
    pub token_input_rate: Decimal,
    pub token_output_rate: Decimal,
    pub context_memory_hourly_rate: Decimal,
    pub context_storage_hourly_rate: Decimal,
    pub gp_memory_hourly_rate: Decimal,
    pub gp_storage_hot_hourly_rate: Decimal,
    pub gp_storage_cold_hourly_rate: Decimal,
}

#[derive(Debug, Deserialize)]
pub struct CapacityConfig {
    pub max_concurrent_requests: i32,
    pub context_memory_capacity_gb: i32,
    pub context_storage_capacity_tb: i32,
    pub gp_memory_capacity_gb: i32,
    pub gp_storage_capacity_tb: i32,
}
