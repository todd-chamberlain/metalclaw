#!/bin/sh
# Distro-agnostic CA certificate installer.
# Resolves the cert from either a local file (build context) or a URL,
# then installs it into the system trust store using whichever trust
# mechanism the base image provides.
#
# Supports: RHEL/Fedora (update-ca-trust), Debian/Ubuntu (update-ca-certificates),
#           Alpine (update-ca-certificates).

set -e

CERT_SRC="/tmp/build-ca-cert.pem"
CA_CERT_URL="${CA_CERT_URL:-}"

# Resolve the cert: prefer local file, fall back to URL
if [ -s "$CERT_SRC" ]; then
    : # cert already in place from COPY
elif [ -n "$CA_CERT_URL" ]; then
    case "$CA_CERT_URL" in
        https://*) ;;
        *)
            echo "ERROR: CA_CERT_URL must use https:// (CA certs over HTTP are vulnerable to MITM)" >&2
            exit 1
            ;;
    esac
    curl -fsSL --proto '=https' "$CA_CERT_URL" -o "$CERT_SRC"
else
    exit 0  # no cert to install
fi

# Install into the correct trust store for this distro
if command -v update-ca-trust >/dev/null 2>&1; then
    # RHEL / Fedora / CentOS
    cp "$CERT_SRC" /etc/pki/ca-trust/source/anchors/custom-ca.crt
    update-ca-trust
elif command -v update-ca-certificates >/dev/null 2>&1; then
    # Debian / Ubuntu / Alpine
    cp "$CERT_SRC" /usr/local/share/ca-certificates/custom-ca.crt
    update-ca-certificates
else
    echo "Warning: no CA trust update tool found, cert installed to /etc/ssl/certs/ only" >&2
    cp "$CERT_SRC" /etc/ssl/certs/custom-ca.crt
fi
