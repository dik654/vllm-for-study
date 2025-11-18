-- Providers table
CREATE TABLE IF NOT EXISTS providers (
    provider_id VARCHAR(64) PRIMARY KEY,
    provider_name VARCHAR(255) NOT NULL,
    endpoint_url VARCHAR(512) NOT NULL,
    provider_token VARCHAR(128) NOT NULL UNIQUE,
    status VARCHAR(32) NOT NULL DEFAULT 'active',

    -- LLM Pricing
    token_input_rate DECIMAL(12, 10) NOT NULL,
    token_output_rate DECIMAL(12, 10) NOT NULL,
    cached_input_discount DECIMAL(3, 2) DEFAULT 0.9,
    request_base_rate DECIMAL(10, 6) NOT NULL,
    batch_discount DECIMAL(3, 2) DEFAULT 0.5,

    -- Context Memory/Storage Pricing
    context_memory_hourly_rate DECIMAL(10, 6) NOT NULL,
    context_storage_hourly_rate DECIMAL(10, 6) NOT NULL,

    -- GP Memory/Storage Pricing
    gp_memory_hourly_rate DECIMAL(10, 6) NOT NULL,
    gp_storage_hot_hourly_rate DECIMAL(10, 6) NOT NULL,
    gp_storage_cold_hourly_rate DECIMAL(10, 6) NOT NULL,

    -- Capacity
    max_concurrent_requests INTEGER NOT NULL,
    context_memory_capacity_gb INTEGER NOT NULL,
    context_storage_capacity_tb INTEGER NOT NULL,
    gp_memory_capacity_gb INTEGER NOT NULL,
    gp_storage_capacity_tb INTEGER NOT NULL,

    -- Metadata
    supported_models JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_provider_status ON providers(status);
CREATE INDEX IF NOT EXISTS idx_provider_created_at ON providers(created_at);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id VARCHAR(64) PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    api_key VARCHAR(128) NOT NULL UNIQUE,
    status VARCHAR(32) NOT NULL DEFAULT 'active',
    tier VARCHAR(32) NOT NULL DEFAULT 'free',

    -- Billing
    billing_email VARCHAR(255),
    payment_method_id VARCHAR(128),
    credit_balance DECIMAL(12, 2) DEFAULT 0.00,

    -- Limits
    rate_limit_per_minute INTEGER DEFAULT 60,
    monthly_budget_usd DECIMAL(10, 2),

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_user_api_key ON users(api_key);
CREATE INDEX IF NOT EXISTS idx_user_status ON users(status);
CREATE INDEX IF NOT EXISTS idx_user_tier ON users(tier);

-- Requests table
CREATE TABLE IF NOT EXISTS requests (
    request_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
    provider_id VARCHAR(64) NOT NULL REFERENCES providers(provider_id),

    -- Request Info
    model VARCHAR(128) NOT NULL,
    endpoint VARCHAR(128) NOT NULL,
    is_batch BOOLEAN DEFAULT FALSE,

    -- Token Usage
    input_tokens INTEGER NOT NULL,
    cached_input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER NOT NULL,
    latency_ms INTEGER NOT NULL,

    -- Cost Breakdown
    provider_cost_usd DECIMAL(12, 8) NOT NULL,
    commission_usd DECIMAL(12, 8) NOT NULL,
    user_charge_usd DECIMAL(12, 8) NOT NULL,

    -- Status
    status VARCHAR(32) NOT NULL,
    error_message TEXT,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_request_user_created ON requests(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_request_provider_created ON requests(provider_id, created_at);
CREATE INDEX IF NOT EXISTS idx_request_created_at ON requests(created_at);
CREATE INDEX IF NOT EXISTS idx_request_is_batch ON requests(is_batch);

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows (
    workflow_id VARCHAR(64) PRIMARY KEY,
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),
    name VARCHAR(255) NOT NULL,
    description TEXT,
    version VARCHAR(16) NOT NULL DEFAULT '1.0.0',

    -- Workflow Definition
    nodes JSONB NOT NULL,
    connections JSONB NOT NULL,
    metadata JSONB NOT NULL,
    settings JSONB NOT NULL,

    is_public BOOLEAN DEFAULT FALSE,
    fork_count INTEGER DEFAULT 0,
    execution_count INTEGER DEFAULT 0,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_workflow_user_created ON workflows(user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_workflow_public_created ON workflows(is_public, created_at);

-- Workflow Executions table
CREATE TABLE IF NOT EXISTS workflow_executions (
    execution_id VARCHAR(64) PRIMARY KEY,
    workflow_id VARCHAR(64) NOT NULL REFERENCES workflows(workflow_id),
    user_id VARCHAR(64) NOT NULL REFERENCES users(user_id),

    state VARCHAR(32) NOT NULL,
    input_data JSONB NOT NULL,
    output_data JSONB,

    progress JSONB,
    node_outputs JSONB,

    started_at TIMESTAMP WITH TIME ZONE NOT NULL,
    completed_at TIMESTAMP WITH TIME ZONE,

    total_duration_ms INTEGER,
    total_cost_usd DECIMAL(12, 8),

    error_message TEXT
);

CREATE INDEX IF NOT EXISTS idx_execution_user_started ON workflow_executions(user_id, started_at);
CREATE INDEX IF NOT EXISTS idx_execution_workflow_started ON workflow_executions(workflow_id, started_at);
CREATE INDEX IF NOT EXISTS idx_execution_state ON workflow_executions(state);
