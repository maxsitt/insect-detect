#!/bin/bash

# Install the insect-detect software including dependencies, required packages and all setup steps

# Source:   https://github.com/maxsitt/insect-detect
# License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
# Author:   Maximilian Sittinger (https://github.com/maxsitt)
# Docs:     https://maxsitt.github.io/insect-detect-docs/

# Immediately exit script on error, undefined variable, or pipe failure
set -euo pipefail

echo "==== Insect Detect Installer ===="
echo

# Install git
if ! command -v git >/dev/null 2>&1; then
    echo "[1/8] Installing git..."
    sudo apt update
    sudo apt install -y git
else
    echo "[1/8] Git is already installed."
fi

# Clone insect-detect repository
cd "$HOME"
if [[ ! -d "insect-detect" ]]; then
    echo
    echo "[2/8] Cloning 'insect-detect' repository..."
    if ! git clone https://github.com/maxsitt/insect-detect; then
        echo "ERROR: Failed to clone repository. Please retry or check your internet connection."
        exit 1
    fi
else
    echo
    echo "[2/8] 'insect-detect' repository already exists."
fi

# Install dependencies for RPi + OAK
cd "$HOME/insect-detect"
echo
echo "[3/8] Installing dependencies for RPi + OAK..."
if ! bash install_dependencies_oak.sh; then
    echo "ERROR: Failed to install dependencies. Please check the error messages above."
    exit 1
fi

# Create virtual environment
cd "$HOME"
if [[ ! -d "env_insdet" ]]; then
    echo
    echo "[4/8] Creating virtual environment 'env_insdet'..."
    if ! python3 -m venv --system-site-packages env_insdet; then
        echo "ERROR: Failed to create virtual environment"
        exit 1
    fi
else
    echo
    echo "[4/8] Virtual environment 'env_insdet' already exists."
fi

# Upgrade pip in the virtual environment
echo
echo "[5/8] Upgrading pip in the virtual environment..."
if ! "$HOME/env_insdet/bin/python3" -m pip install --upgrade pip; then
    echo "ERROR: Failed to upgrade pip"
    exit 1
fi

# Install required packages in the virtual environment
cd "$HOME/insect-detect"
echo
echo "[6/8] Installing required Python packages..."
if ! "$HOME/env_insdet/bin/python3" -m pip install --upgrade -r requirements.txt; then
    echo "ERROR: Failed to install required Python packages"
    exit 1
fi

# Generate self-signed SSL certificates
echo
echo "[7/8] Generating self-signed SSL certificates for web app HTTPS support..."
if ! bash generate_ssl_certificates.sh; then
    echo "WARNING: Failed to generate SSL certificates"
    echo "The web app will still work over HTTP, but HTTPS won't be available."
fi

# Set up systemd service
echo
echo "[8/8] Installing systemd service for automatic startup..."
if ! sudo cp insect-detect-startup.service /etc/systemd/system/; then
    echo "ERROR: Failed to copy service file"
    exit 1
fi
if ! sudo systemctl daemon-reload; then
    echo "ERROR: Failed to reload systemd daemon"
    exit 1
fi
if ! sudo systemctl enable insect-detect-startup.service; then
    echo "ERROR: Failed to enable startup service"
    exit 1
fi

echo
echo "Installation complete!"
echo
echo "Use the web app to configure your custom settings."
echo "...or modify the 'insect-detect/configs/config_custom.yaml' file directly."
echo "The startup settings will be active after the next reboot."
echo
echo "You can now run the web app with:"
echo "env_insdet/bin/python3 insect-detect/webapp.py"
