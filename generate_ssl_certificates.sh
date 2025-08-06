#!/bin/bash

# Generate self-signed SSL certificates to enable HTTPS for the Insect Detect web app
# HTTPS is required for the browser Geolocation API to get GPS coordinates

# Source:   https://github.com/maxsitt/insect-detect
# License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
# Author:   Maximilian Sittinger (https://github.com/maxsitt)
# Docs:     https://maxsitt.github.io/insect-detect-docs/

# Immediately exit script on error, undefined variable, or pipe failure
set -euo pipefail

HOSTNAME=$(hostname)
CERT_DAYS=1825  # number of days the certificates are valid (default: 5 years)
SSL_DIR="$HOME/ssl_certificates"

# Create SSL certificates directory
mkdir -p "$SSL_DIR" 2>/dev/null || { echo "ERROR: Could not create '$SSL_DIR'"; exit 1; }

# Generate self-signed SSL certificates
cd "$SSL_DIR"
echo "Generating self-signed SSL certificates..."
if ! openssl req -x509 -newkey rsa:2048 -sha256 -nodes -out cert.pem -keyout key.pem -days "$CERT_DAYS" \
    -subj "/CN=${HOSTNAME}" \
    -addext "subjectAltName=DNS:${HOSTNAME},DNS:${HOSTNAME}.local" 2>/dev/null; then
    echo "ERROR: Failed to generate SSL certificates"
    exit 1
fi

# Set permissions for the key file
chmod 600 key.pem 2>/dev/null || echo "WARNING: Failed to set proper permissions on private key"

# Verify that the certificates were created successfully
if [[ -f "cert.pem" && -f "key.pem" ]]; then
    echo "SSL certificates generated successfully! They are located in $SSL_DIR"
    echo "Valid until: $(openssl x509 -in cert.pem -noout -enddate | cut -d= -f2)"
else
  echo "ERROR: Failed to generate SSL certificates - files not found."
  exit 1
fi
