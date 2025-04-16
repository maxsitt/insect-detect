#!/bin/bash

# Generate self-signed SSL certificates to enable HTTPS for the Insect Detect web app
# HTTPS is required for browser Geolocation API to get GPS coordinates

# Source:   https://github.com/maxsitt/insect-detect
# License:  GNU GPLv3 (https://choosealicense.com/licenses/gpl-3.0/)
# Author:   Maximilian Sittinger (https://github.com/maxsitt)
# Docs:     https://maxsitt.github.io/insect-detect-docs/

HOSTNAME=$(hostname)
CERT_DAYS=1825  # number of days the certificates are valid (default: 5 years)

mkdir -p ~/ssl_certificates
cd ~/ssl_certificates

openssl req -x509 -newkey rsa:2048 -sha256 -nodes -out cert.pem -keyout key.pem -days $CERT_DAYS \
  -subj "/CN=${HOSTNAME}" \
  -addext "subjectAltName=DNS:${HOSTNAME},DNS:${HOSTNAME}.local"

chmod 600 key.pem

if [ -f "cert.pem" ] && [ -f "key.pem" ]; then
  echo "SSL certificates generated successfully at ~/ssl_certificates/"
  echo "Valid until: $(openssl x509 -in cert.pem -noout -enddate | cut -d= -f2)"
else
  echo "Error: SSL certificate generation failed"
fi

cd ..
