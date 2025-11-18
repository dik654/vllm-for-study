# Distributed LLM Platform

A high-performance distributed LLM platform built with Rust, featuring OpenRouter-style API routing and n8n-style workflow orchestration.

## Architecture

This platform consists of three main components:

1. **Router** - Central API gateway and request router
2. **Provider** - LLM provider agent that interfaces with vLLM
3. **Shared** - Common models, configuration, and utilities

### Technology Stack

- **Language**: Rust 1.75+
- **Web Framework**: Axum 0.7
- **Database**: PostgreSQL 15 with SQLx
- **Caching**: Redis 7
- **Async Runtime**: Tokio
- **Monitoring**: Prometheus + Grafana

## Features

### Router Service
- Provider management (create, list, get)
- User management and API key generation
- Request logging and analytics
- LLM API proxy (OpenAI-compatible)
- Workflow management (CRUD operations)
- Workflow execution tracking

### Provider Service
- vLLM integration
- Chat completions endpoint
- Health monitoring
- Automatic request forwarding

### Shared Library
- Database models with SQLx
- Configuration management
- Error handling
- Type-safe decimal pricing

## Getting Started

### Prerequisites

- Rust 1.75 or later
- Docker and Docker Compose
- PostgreSQL 15
- Redis 7

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd distributed-llm-platform
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Start the services:
```bash
make docker-up
```

4. Run database migrations:
```bash
export DATABASE_URL="postgresql://postgres:password@localhost:5432/distributed_llm"
make db-upgrade
```

### Development

Build the workspace:
```bash
make build
```

Run tests:
```bash
make test
```

Run router locally:
```bash
make run-router
```

Run provider locally:
```bash
make run-provider
```

## API Endpoints

### Router (Port 8000)

#### Health & Info
- `GET /` - API information
- `GET /health` - Health check

#### Providers
- `POST /v1/providers` - Create provider
- `GET /v1/providers` - List providers
- `GET /v1/providers/:id` - Get provider

#### Users
- `POST /v1/users` - Create user
- `GET /v1/users` - List users
- `GET /v1/users/:id` - Get user
- `GET /v1/users/:id/requests` - List user requests

#### LLM API
- `POST /v1/chat/completions` - Chat completions (OpenAI-compatible)

#### Workflows
- `POST /v1/users/:user_id/workflows` - Create workflow
- `GET /v1/users/:user_id/workflows` - List workflows
- `GET /v1/workflows/:id` - Get workflow
- `PUT /v1/workflows/:id` - Update workflow
- `DELETE /v1/workflows/:id` - Delete workflow
- `POST /v1/workflows/:id/execute/:user_id` - Execute workflow
- `GET /v1/executions/:id` - Get execution status

### Provider (Port 8001)

- `GET /` - Provider information
- `GET /health` - Provider health
- `GET /vllm/health` - vLLM health check
- `POST /v1/chat/completions` - Chat completions

## Environment Variables

### Router
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/distributed_llm
REDIS_URL=redis://localhost:6379
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
```

### Provider
```env
PROVIDER_ID=provider_001
PROVIDER_NAME=Provider 01
VLLM_HOST=localhost
VLLM_PORT=8000
HOST=0.0.0.0
PORT=8001
DATABASE_URL=postgresql://postgres:password@localhost:5432/distributed_llm
REDIS_URL=redis://localhost:6379
LOG_LEVEL=info
```

## Database Schema

### Tables

- **providers** - LLM provider configurations with pricing
- **users** - User accounts and API keys
- **requests** - Request logs with token usage and costs
- **workflows** - n8n-style workflow definitions
- **workflow_executions** - Workflow execution history

See `migrations/20250118000001_initial_schema.sql` for full schema.

## Docker Deployment

Build and run with Docker Compose:

```bash
# Build images
make docker-build

# Start all services
make docker-up

# View logs
make docker-logs

# Stop services
make docker-down
```

Services:
- Router: http://localhost:8000
- Provider: http://localhost:8001
- PostgreSQL: localhost:5432
- Redis: localhost:6379
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000

## Development Workflow

1. Make changes to the code
2. Test locally: `cargo test`
3. Build: `cargo build`
4. Format code: `make format`
5. Lint: `make lint`
6. Commit and push

## Monitoring

Access Grafana at http://localhost:3000 (admin/admin) to view:
- Request rates
- Latency metrics
- Error rates
- Provider health

Prometheus metrics available at:
- Router: http://localhost:8000/metrics
- Provider: http://localhost:8001/metrics

## Project Structure

```
distributed-llm-platform/
├── router/              # Central router service
│   ├── src/
│   │   ├── main.rs     # Entry point
│   │   ├── handlers.rs # API handlers
│   │   └── db.rs       # Database state
│   └── Cargo.toml
├── provider/            # Provider agent service
│   ├── src/
│   │   ├── main.rs     # Entry point
│   │   ├── handlers.rs # API handlers
│   │   └── vllm_client.rs # vLLM integration
│   └── Cargo.toml
├── shared/              # Shared library
│   ├── src/
│   │   ├── lib.rs      # Module exports
│   │   ├── models.rs   # Database models
│   │   ├── config.rs   # Configuration
│   │   └── errors.rs   # Error types
│   └── Cargo.toml
├── migrations/          # SQL migrations
├── docker/              # Docker configurations
├── examples/            # Example workflows
└── Cargo.toml          # Workspace definition
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linting
5. Submit a pull request

## License

[Your License Here]

## References

- [Axum Documentation](https://docs.rs/axum)
- [SQLx Documentation](https://docs.rs/sqlx)
- [vLLM Documentation](https://docs.vllm.ai)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
