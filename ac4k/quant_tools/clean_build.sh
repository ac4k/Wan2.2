#!/bin/bash
# Clean build directory and reinstall

set -e

echo "Cleaning build directory..."
rm -rf dist/
rm -rf *.egg-info/
find . -name "*.so" -type f -delete
find . -name "*.cpython-*.so" -type f -delete

echo "Build directory cleaned!"
echo ""
echo "Now run: pip install -e . --no-deps"

