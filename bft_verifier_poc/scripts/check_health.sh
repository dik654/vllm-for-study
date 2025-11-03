#!/bin/bash
# Health check script for BFT Verifier PoC

set -e

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "========================================="
echo "  BFT Verifier - Health Check"
echo "========================================="
echo ""

# Check if Docker Compose is running
if ! docker-compose ps > /dev/null 2>&1; then
    echo -e "${RED}✗ Docker Compose not available${NC}"
    exit 1
fi

# Check each service
echo "=== Service Status ==="

check_service() {
    local service=$1
    local port=$2

    if docker-compose ps | grep $service | grep -q "Up"; then
        echo -e "${GREEN}✓ $service is running${NC}"

        # Check if port is accessible
        if [ ! -z "$port" ]; then
            CONTAINER_NAME=$(docker-compose ps -q $service)
            if docker exec $CONTAINER_NAME nc -z localhost $port 2>/dev/null; then
                echo -e "  ${GREEN}✓ Port $port is open${NC}"
            else
                echo -e "  ${YELLOW}⚠ Port $port check failed${NC}"
            fi
        fi
        return 0
    else
        echo -e "${RED}✗ $service is not running${NC}"
        return 1
    fi
}

# Check all services
HEALTHY=0
check_service "coordinator" "50051" && HEALTHY=$((HEALTHY + 1))
check_service "verifier-1" "50052" && HEALTHY=$((HEALTHY + 1))
check_service "verifier-2" "50053" && HEALTHY=$((HEALTHY + 1))
check_service "verifier-3" "50054" && HEALTHY=$((HEALTHY + 1))
check_service "verifier-4" "50055" && HEALTHY=$((HEALTHY + 1))
check_service "agent" "" && HEALTHY=$((HEALTHY + 1))

echo ""
echo "=== Recent Activity ==="

# Check for recent consensus results
RECENT_CONSENSUS=$(docker-compose logs coordinator 2>/dev/null | \
                   grep "Consensus complete" | tail -3)

if [ -z "$RECENT_CONSENSUS" ]; then
    echo -e "${YELLOW}⚠ No recent consensus activity${NC}"
else
    echo -e "${GREEN}✓ Recent consensus results:${NC}"
    echo "$RECENT_CONSENSUS" | while read line; do
        echo "  $line"
    done
fi

echo ""
echo "=== Summary ==="
echo "Healthy services: $HEALTHY/6"

if [ $HEALTHY -eq 6 ]; then
    echo -e "${GREEN}✓ All services are healthy${NC}"
    exit 0
elif [ $HEALTHY -ge 4 ]; then
    echo -e "${YELLOW}⚠ Some services are down${NC}"
    exit 1
else
    echo -e "${RED}✗ System is unhealthy${NC}"
    exit 1
fi
