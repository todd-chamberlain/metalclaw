#!/bin/sh
# Distro-agnostic system package installer.
# Detects the base image's package manager and installs packages.
#
# Usage:
#   install-system-deps.sh              # install core deps (Vulkan, build tools, runtimes)
#   install-system-deps.sh pkg1 pkg2    # install only the listed packages
#
# Supported package managers (6 families):
#   dnf / microdnf  — Fedora, RHEL 8+, Rocky, Alma, CentOS Stream, UBI
#   tdnf            — Azure Linux / CBL-Mariner
#   yum             — RHEL 7, CentOS 7 (legacy)
#   apt-get         — Debian, Ubuntu, and all derivatives
#   zypper          — openSUSE, SLES, MicroOS
#   apk             — Alpine, Wolfi, Chainguard

set -e

# ── Detect package manager (order matters: dnf before yum) ────────────
if command -v dnf >/dev/null 2>&1; then
    PKG_MGR="dnf"
elif command -v microdnf >/dev/null 2>&1; then
    PKG_MGR="microdnf"
elif command -v tdnf >/dev/null 2>&1; then
    PKG_MGR="tdnf"
elif command -v yum >/dev/null 2>&1; then
    PKG_MGR="yum"
elif command -v apt-get >/dev/null 2>&1; then
    PKG_MGR="apt-get"
elif command -v zypper >/dev/null 2>&1; then
    PKG_MGR="zypper"
elif command -v apk >/dev/null 2>&1; then
    PKG_MGR="apk"
else
    echo "ERROR: No supported package manager found (need dnf/apt-get/apk/zypper/tdnf)" >&2
    exit 1
fi

echo "Detected package manager: $PKG_MGR"

# ── Resolve package list ──────────────────────────────────────────────
# If arguments are provided, install those (extra packages mode).
# Otherwise install the full core dependency set for this distro.
if [ $# -gt 0 ]; then
    PKGS="$*"
else
    case "$PKG_MGR" in
        dnf|microdnf)
            PKGS="mesa-vulkan-drivers vulkan-loader vulkan-tools vulkan-headers vulkan-loader-devel glslang glslc cmake gcc-c++ git python3 python3-pip curl ca-certificates nodejs npm bash"
            ;;
        tdnf)
            # Note: Azure Linux may not ship all Vulkan packages; mesa/vulkan-tools
            # availability depends on the specific Azure Linux version and repos.
            PKGS="mesa-vulkan-drivers vulkan-loader vulkan-tools vulkan-headers cmake gcc-c++ git python3 python3-pip curl ca-certificates nodejs npm bash"
            ;;
        yum)
            PKGS="mesa-vulkan-drivers vulkan-loader vulkan-tools vulkan-headers vulkan-loader-devel glslang cmake gcc-c++ git python3 python3-pip curl ca-certificates nodejs npm bash"
            ;;
        apt-get)
            PKGS="libvulkan1 libvulkan-dev mesa-vulkan-drivers vulkan-tools glslang-tools spirv-tools cmake g++ git python3 python3-pip curl ca-certificates nodejs npm bash"
            ;;
        zypper)
            PKGS="Mesa-vulkan-drivers libvulkan1 vulkan-devel vulkan-tools glslang-devel cmake gcc-c++ git python3 python3-pip curl ca-certificates nodejs npm bash"
            ;;
        apk)
            PKGS="vulkan-loader vulkan-loader-dev vulkan-headers vulkan-tools glslang cmake g++ git make python3 py3-pip curl ca-certificates nodejs npm bash"
            ;;
    esac
fi

# ── Install ───────────────────────────────────────────────────────────
case "$PKG_MGR" in
    dnf|microdnf|tdnf)
        "$PKG_MGR" install -y $PKGS && "$PKG_MGR" clean all
        ;;
    yum)
        yum install -y $PKGS && yum clean all
        ;;
    apt-get)
        apt-get update && apt-get install -y --no-install-recommends $PKGS && rm -rf /var/lib/apt/lists/*
        ;;
    zypper)
        zypper --non-interactive install $PKGS && zypper clean --all
        ;;
    apk)
        apk add --no-cache $PKGS
        ;;
esac

echo "Packages installed via $PKG_MGR"
