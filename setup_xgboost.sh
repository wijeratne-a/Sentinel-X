#!/bin/bash
# Setup script for Sentinel-X XGBoost dependencies on macOS

echo "=========================================="
echo "Sentinel-X: XGBoost Setup for macOS"
echo "=========================================="
echo ""

# Check if Homebrew is installed
if ! command -v brew &> /dev/null; then
    echo "Homebrew is not installed."
    echo ""
    echo "To install Homebrew, run this command in your terminal:"
    echo '  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
    echo ""
    echo "After installing Homebrew, run this script again."
    exit 1
fi

echo "[OK] Homebrew is installed"
echo ""

# Install libomp
echo "Installing libomp (OpenMP runtime)..."
brew install libomp

if [ $? -eq 0 ]; then
    echo ""
    echo "[OK] libomp installed successfully"
    echo ""
    echo "Reinstalling XGBoost..."
    pip uninstall -y xgboost
    pip install xgboost
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "=========================================="
        echo "[OK] Setup complete! XGBoost should now work."
        echo "=========================================="
    else
        echo ""
        echo "Error: Failed to reinstall XGBoost"
        exit 1
    fi
else
    echo ""
    echo "Error: Failed to install libomp"
    exit 1
fi
