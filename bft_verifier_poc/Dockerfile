# Multi-stage build for BFT Verifier PoC
# Stage 1: Builder
FROM rust:1.75-slim as builder

WORKDIR /app

# Install dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml ./
COPY common/ ./common/
COPY coordinator/ ./coordinator/
COPY verifier/ ./verifier/
COPY agent/ ./agent/
COPY proto/ ./proto/

# Build all binaries in release mode
RUN cargo build --release --workspace

# Stage 2: Runtime
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Copy binaries from builder
COPY --from=builder /app/target/release/coordinator /usr/local/bin/
COPY --from=builder /app/target/release/verifier /usr/local/bin/
COPY --from=builder /app/target/release/agent /usr/local/bin/

# Default command (can be overridden)
CMD ["coordinator"]
