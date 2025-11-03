#!/bin/bash

# BFT Verifier PoC - Quick Start Script
# This script helps you quickly start the BFT verification system

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  BFT Verifier PoC - Quick Start${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed${NC}"
    echo "Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}Error: Docker Compose is not installed${NC}"
    echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
    exit 1
fi

echo -e "${GREEN}✓ Docker and Docker Compose are installed${NC}"
echo ""

# Show menu
echo "Choose an option:"
echo "  1) Start full system (Docker Compose)"
echo "  2) Build Rust binaries locally"
echo "  3) Run tests"
echo "  4) Clean up Docker containers"
echo "  5) View logs"
echo "  6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo ""
        echo -e "${BLUE}Starting full BFT system with Docker Compose...${NC}"
        echo "This will start:"
        echo "  - 1 Coordinator (port 50051)"
        echo "  - 4 Verifiers (ports 50052-50055)"
        echo "  - 1 Agent (submitting metrics every 5 seconds)"
        echo ""

        docker-compose up --build
        ;;

    2)
        echo ""
        echo -e "${BLUE}Building Rust binaries locally...${NC}"
        echo ""

        # Check if Rust is installed
        if ! command -v cargo &> /dev/null; then
            echo -e "${RED}Error: Rust is not installed${NC}"
            echo "Install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            exit 1
        fi

        # Check if protobuf compiler is installed
        if ! command -v protoc &> /dev/null; then
            echo -e "${YELLOW}Warning: protobuf compiler not found${NC}"
            echo "Install it with:"
            echo "  Ubuntu/Debian: sudo apt-get install protobuf-compiler"
            echo "  macOS: brew install protobuf"
            echo ""
            read -p "Continue anyway? [y/N]: " continue
            if [[ ! $continue =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi

        echo "Building all workspace members..."
        cargo build --release --workspace

        echo ""
        echo -e "${GREEN}✓ Build complete!${NC}"
        echo ""
        echo "Binaries are in:"
        echo "  ./target/release/coordinator"
        echo "  ./target/release/verifier"
        echo "  ./target/release/agent"
        echo ""
        echo "To run:"
        echo "  Terminal 1: RUST_LOG=info ./target/release/coordinator"
        echo "  Terminal 2: RUST_LOG=info VERIFIER_ID=verifier-1 LISTEN_ADDR=0.0.0.0:50052 ./target/release/verifier"
        echo "  Terminal 3: RUST_LOG=info AGENT_ID=agent-1 MODE=single REQUEST_COUNT=5 ./target/release/agent"
        ;;

    3)
        echo ""
        echo -e "${BLUE}Running tests...${NC}"
        echo ""

        if ! command -v cargo &> /dev/null; then
            echo -e "${RED}Error: Rust is not installed${NC}"
            exit 1
        fi

        cargo test --workspace --verbose

        echo ""
        echo -e "${GREEN}✓ All tests passed!${NC}"
        ;;

    4)
        echo ""
        echo -e "${BLUE}Cleaning up Docker containers...${NC}"
        echo ""

        docker-compose down -v

        echo ""
        echo -e "${GREEN}✓ Cleanup complete!${NC}"
        ;;

    5)
        echo ""
        echo -e "${BLUE}Viewing logs...${NC}"
        echo ""
        echo "Choose component:"
        echo "  1) All"
        echo "  2) Coordinator"
        echo "  3) Verifier-1"
        echo "  4) Agent"
        echo ""
        read -p "Enter choice [1-4]: " log_choice

        case $log_choice in
            1) docker-compose logs -f ;;
            2) docker-compose logs -f coordinator ;;
            3) docker-compose logs -f verifier-1 ;;
            4) docker-compose logs -f agent ;;
            *) echo "Invalid choice" ;;
        esac
        ;;

    6)
        echo "Exiting..."
        exit 0
        ;;

    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac
