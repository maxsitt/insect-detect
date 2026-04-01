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

# Set environment variable telling OpenBLAS to use ARM Cortex-A53 optimized code paths
if ! grep -q "OPENBLAS_CORETYPE" ~/.bashrc; then
    echo "Setting OPENBLAS_CORETYPE=ARMV8 environment variable..."
    echo "export OPENBLAS_CORETYPE=ARMV8" >> ~/.bashrc
else
    echo "OPENBLAS_CORETYPE environment variable is already set."
fi
export OPENBLAS_CORETYPE=ARMV8
echo

# Set up udev rules to allow non-root access to OAK camera
echo "Setting up udev rules for OAK camera access..."
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
echo

# Install git
if ! command -v git >/dev/null 2>&1; then
    echo "[1/5] Installing git..."
    sudo apt update
    sudo apt install -y git
else
    echo "[1/5] Git is already installed."
fi

# Install uv package manager (https://docs.astral.sh/uv)
if ! command -v uv >/dev/null 2>&1; then
    echo
    echo "[2/5] Installing uv..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "ERROR: Failed to install uv. Please check the error messages above."
        exit 1
    fi
    # Make uv available in the current shell session
    export PATH="$HOME/.local/bin:$PATH"
else
    echo
    echo "[2/5] uv is already installed."
fi

# Clone insect-detect repository into the home directory
cd "$HOME"
if [[ ! -d "insect-detect" ]]; then
    echo
    echo "[3/5] Cloning 'insect-detect' repository..."
    if ! git clone https://github.com/maxsitt/insect-detect; then
        echo "ERROR: Failed to clone repository. Please retry or check your internet connection."
        exit 1
    fi
else
    echo
    echo "[3/5] 'insect-detect' repository already exists."
fi

# All remaining steps are run from the insect-detect repository directory
cd "$HOME/insect-detect"

# Create virtual environment with access to system site packages (required for GPIO access)
echo
echo "[4/5] Creating virtual environment..."
if ! uv venv --system-site-packages; then
    echo "ERROR: Failed to create virtual environment."
    exit 1
fi

# Install required Python packages into the virtual environment
echo
echo "[5/5] Installing required Python packages..."
if ! uv sync; then
    echo "ERROR: Failed to install required Python packages."
    exit 1
fi

# Download detection models defined in models/models.json
echo
echo "[+] Downloading detection models..."
if ! uv run python models/download_models.py; then
    echo "WARNING: Failed to download one or more detection models."
    echo "You can retry manually by running 'uv run python models/download_models.py'."
fi

# Generate self-signed SSL certificates for web app HTTPS support
echo
echo "[+] Generating self-signed SSL certificates for web app HTTPS support..."
if ! bash generate_ssl_certificates.sh; then
    echo "WARNING: Failed to generate SSL certificates."
    echo "The web app will still work over HTTP, but HTTPS won't be available."
fi

# Install and enable systemd service for automatic startup on boot
echo
echo "[+] Installing systemd service for automatic startup..."
if ! sudo cp insect-detect-startup.service /etc/systemd/system/; then
    echo "ERROR: Failed to copy service file."
    exit 1
fi
if ! sudo systemctl daemon-reload; then
    echo "ERROR: Failed to reload systemd daemon."
    exit 1
fi
if ! sudo systemctl enable insect-detect-startup.service; then
    echo "ERROR: Failed to enable startup service."
    exit 1
fi

echo
echo "Installation complete!"
echo
echo "Use the web app to configure your custom settings,"
echo "or modify the 'insect-detect/configs/config.yaml' file directly."
echo "The automatic startup service will be active after the next reboot."
echo
echo "To run the scripts manually, first navigate into the insect-detect directory:"
echo "  cd insect-detect"
echo
echo "Then run the scripts with:"
echo "  uv run webapp"
echo "  uv run capture"
echo
echo "NOTE: Please close and reopen your terminal before using 'uv',"
echo "to ensure the PATH is updated and 'uv' is available."
