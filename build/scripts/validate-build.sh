#!/bin/bash
# Build validation script

set -euo pipefail

echo "🔍 Validating build artifacts..."

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if dist directory exists
if [ ! -d "dist" ]; then
    print_error "dist/ directory not found. Run 'make build' first."
    exit 1
fi

# Check for wheel file
if ls dist/*.whl 1> /dev/null 2>&1; then
    print_status "Wheel file found"
    wheel_file=$(ls dist/*.whl | head -1)
    echo "  Wheel: $(basename "$wheel_file")"
else
    print_error "No wheel file found in dist/"
    exit 1
fi

# Check for source distribution
if ls dist/*.tar.gz 1> /dev/null 2>&1; then
    print_status "Source distribution found"
    sdist_file=$(ls dist/*.tar.gz | head -1)
    echo "  Sdist: $(basename "$sdist_file")"
else
    print_error "No source distribution found in dist/"
    exit 1
fi

# Validate wheel contents
echo "🔍 Validating wheel contents..."
python -m zipfile -l "$wheel_file" | grep -q "quantum_planner" && print_status "Package contents found in wheel"

# Check wheel metadata
echo "🔍 Checking wheel metadata..."
python -m wheel tags "$wheel_file" && print_status "Wheel tags are valid"

# Validate with twine
if command -v twine &> /dev/null; then
    echo "🔍 Running twine check..."
    twine check dist/* && print_status "Twine validation passed"
else
    print_warning "twine not found, skipping validation"
fi

# Check package size
wheel_size=$(stat -c%s "$wheel_file" 2>/dev/null || stat -f%z "$wheel_file" 2>/dev/null)
wheel_size_mb=$((wheel_size / 1024 / 1024))

if [ "$wheel_size_mb" -gt 50 ]; then
    print_warning "Wheel file is large (${wheel_size_mb}MB). Consider optimization."
else
    print_status "Wheel size is reasonable (${wheel_size_mb}MB)"
fi

# Test installation in temporary environment
echo "🔍 Testing installation..."
temp_venv=$(mktemp -d)
python -m venv "$temp_venv"
source "$temp_venv/bin/activate"

pip install --quiet "$wheel_file"
python -c "import quantum_planner; print(f'Import successful: {quantum_planner.__version__}')" && print_status "Package imports successfully"

# Cleanup
deactivate
rm -rf "$temp_venv"

print_status "Build validation completed successfully!"

echo ""
echo "📦 Build artifacts:"
echo "  - Wheel: $(basename "$wheel_file") (${wheel_size_mb}MB)"
echo "  - Source: $(basename "$sdist_file")"
echo ""
echo "Ready for release! 🚀"