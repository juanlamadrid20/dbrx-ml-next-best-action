#!/bin/bash

# Sync requirements.txt from pyproject.toml for Databricks compatibility
# This script generates requirements.txt from UV's dependency resolution

set -e

echo "🔄 Syncing requirements.txt from pyproject.toml..."

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ UV not found. Please install UV first:"
    echo "   curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Generate requirements.txt from pyproject.toml
echo "📦 Generating requirements.txt from pyproject.toml dependencies..."

# Export main dependencies
uv pip compile pyproject.toml --output-file requirements.txt

# Add header comment to requirements.txt (create temporary file for cross-platform compatibility)
{
    echo "# This file is auto-generated from pyproject.toml by sync_requirements.sh"
    echo "# DO NOT EDIT MANUALLY - Edit pyproject.toml instead and run: ./sync_requirements.sh"
    echo "# Generated on: $(date)"
    echo "#"
    cat requirements.txt
} > requirements.txt.tmp && mv requirements.txt.tmp requirements.txt

echo "✅ requirements.txt updated successfully!"
echo "📄 Dependencies exported for Databricks notebook compatibility"

# Show summary
echo ""
echo "📊 Summary:"
echo "   Dependencies in pyproject.toml: $(grep -c '^\s*".*>=.*",$' pyproject.toml || echo '0')"
echo "   Dependencies in requirements.txt: $(grep -c '^[^#].*==' requirements.txt || echo '0')"

echo ""
echo "💡 Usage:"
echo "   - Local development: uv pip sync"
echo "   - Databricks notebooks: %pip install -r ../requirements.txt"