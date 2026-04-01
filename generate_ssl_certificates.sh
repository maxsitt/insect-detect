#!/bin/bash

# Generate self-signed SSL certificates to enable HTTPS for the Insect Detect web app
# HTTPS is required for the browser Geolocation API to get GPS coordinates

# Source:   https://github.com/maxsitt/insect-detect
# License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
# Author:   Maximilian Sittinger (https://github.com/maxsitt)
# Docs:     https://maxsitt.github.io/insect-detect-docs/

# Immediately exit script on error, undefined variable, or pipe failure
set -euo pipefail

DEVICE_HOSTNAME=$(hostname)
CERT_DAYS=1825  # number of days the certificates are valid (default: 5 years)
SSL_DIR="$HOME/insect-detect/ssl"

# Create SSL certificates directory
mkdir -p "$SSL_DIR" || { echo "ERROR: Could not create '$SSL_DIR'."; exit 1; }

# Generate self-signed SSL certificates
echo "Generating self-signed SSL certificates..."
if ! openssl req -x509 -newkey rsa:2048 -sha256 -nodes \
    -out "$SSL_DIR/cert.pem" -keyout "$SSL_DIR/key.pem" \
    -days "$CERT_DAYS" \
    -subj "/CN=${DEVICE_HOSTNAME}" \
    -addext "subjectAltName=DNS:${DEVICE_HOSTNAME},DNS:${DEVICE_HOSTNAME}.local" 2>/dev/null; then
    echo "ERROR: Failed to generate SSL certificates."
    exit 1
fi

# Set permissions for the key file
chmod 600 "$SSL_DIR/key.pem" || echo "WARNING: Failed to set proper permissions on private key."

# Verify that the certificates were created successfully
if [[ -f "$SSL_DIR/cert.pem" && -f "$SSL_DIR/key.pem" ]]; then
    echo "SSL certificates generated successfully! They are located in $SSL_DIR"
    echo "Valid until: $(openssl x509 -in "$SSL_DIR/cert.pem" -noout -enddate | cut -d= -f2)"
else
    echo "ERROR: Failed to generate SSL certificates - files not found."
    exit 1
fi
