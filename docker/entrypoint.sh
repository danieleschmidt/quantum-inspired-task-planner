#!/bin/bash
set -e

# Quantum Task Planner Docker Entrypoint Script

# Environment variables with defaults
export QUANTUM_PLANNER_ENV=${QUANTUM_PLANNER_ENV:-production}
export LOG_LEVEL=${LOG_LEVEL:-INFO}
export BACKEND_TYPE=${BACKEND_TYPE:-auto}
export CONFIG_FILE=${CONFIG_FILE:-/app/config/production.yaml}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to wait for a service to be ready
wait_for_service() {
    local host=$1
    local port=$2
    local timeout=${3:-30}
    
    log "Waiting for $host:$port to be ready..."
    
    local count=0
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $count -ge $timeout ]; then
            error "Timeout waiting for $host:$port"
            exit 1
        fi
        sleep 1
        count=$((count + 1))
    done
    
    log "$host:$port is ready"
}

# Function to validate environment
validate_environment() {
    log "Validating environment..."
    
    # Check required directories exist
    for dir in /app/data /app/logs /app/config; do
        if [ ! -d "$dir" ]; then
            warn "Directory $dir does not exist, creating..."
            mkdir -p "$dir"
        fi
    done
    
    # Check configuration file
    if [ ! -f "$CONFIG_FILE" ]; then
        warn "Configuration file $CONFIG_FILE not found, using defaults"
    fi
    
    # Validate quantum credentials if backend requires them
    if [ "$BACKEND_TYPE" = "dwave" ] && [ -z "$DWAVE_API_TOKEN" ]; then
        warn "D-Wave backend selected but DWAVE_API_TOKEN not set"
    fi
    
    if [ "$BACKEND_TYPE" = "azure" ] && [ -z "$AZURE_QUANTUM_RESOURCE_ID" ]; then
        warn "Azure Quantum backend selected but AZURE_QUANTUM_RESOURCE_ID not set"
    fi
    
    log "Environment validation complete"
}

# Function to perform health check
health_check() {
    log "Performing health check..."
    
    # Check if quantum-planner command is available
    if ! command_exists quantum-planner; then
        error "quantum-planner command not found"
        exit 1
    fi
    
    # Test basic functionality
    if ! quantum-planner --version >/dev/null 2>&1; then
        error "quantum-planner version check failed"
        exit 1
    fi
    
    log "Health check passed"
}

# Function to setup logging
setup_logging() {
    log "Setting up logging..."
    
    # Create log directory if it doesn't exist
    mkdir -p /app/logs
    
    # Set up log rotation if logrotate is available
    if command_exists logrotate; then
        cat > /etc/logrotate.d/quantum-planner << EOF
/app/logs/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 quantum quantum
}
EOF
    fi
    
    log "Logging setup complete"
}

# Function to handle graceful shutdown
cleanup() {
    log "Received shutdown signal, cleaning up..."
    
    # Kill any background processes
    jobs -p | xargs -r kill
    
    # Cleanup temporary files
    rm -rf /tmp/quantum-planner-*
    
    log "Cleanup complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Main initialization
main() {
    log "Starting Quantum Task Planner (Environment: $QUANTUM_PLANNER_ENV)"
    
    # Validate environment
    validate_environment
    
    # Setup logging
    setup_logging
    
    # Perform health check
    health_check
    
    # Wait for external services if configured
    if [ -n "$REDIS_HOST" ] && [ -n "$REDIS_PORT" ]; then
        wait_for_service "$REDIS_HOST" "$REDIS_PORT"
    fi
    
    if [ -n "$POSTGRES_HOST" ] && [ -n "$POSTGRES_PORT" ]; then
        wait_for_service "$POSTGRES_HOST" "$POSTGRES_PORT"
    fi
    
    log "Initialization complete"
    
    # Execute the command passed to the container
    if [ $# -eq 0 ]; then
        # No command provided, show help
        log "No command provided, showing help..."
        exec quantum-planner --help
    elif [ "$1" = "server" ]; then
        # Start the server
        log "Starting Quantum Task Planner server..."
        exec quantum-planner server \
            --host 0.0.0.0 \
            --port 8000 \
            --config "$CONFIG_FILE" \
            --log-level "$LOG_LEVEL"
    elif [ "$1" = "worker" ]; then
        # Start a worker process
        log "Starting Quantum Task Planner worker..."
        exec quantum-planner worker \
            --config "$CONFIG_FILE" \
            --log-level "$LOG_LEVEL"
    elif [ "$1" = "shell" ]; then
        # Start interactive shell
        log "Starting interactive shell..."
        exec /bin/bash
    elif [ "$1" = "test" ]; then
        # Run tests
        log "Running tests..."
        exec python -m pytest tests/ -v
    elif [ "$1" = "benchmark" ]; then
        # Run benchmarks
        log "Running benchmarks..."
        exec python -m pytest benchmarks/ -v --benchmark-only
    else
        # Execute the provided command
        log "Executing command: $*"
        exec "$@"
    fi
}

# Run main function
main "$@"