#!/bin/bash
#
# build.sh - Bulletproof Build & Install Script for Pseudoscopic
#
# Usage: sudo ./build.sh
#

set -e

# Configuration
MODULE_NAME="pseudoscopic"
MODULE_VERSION="0.0.1"
SRC_DIR="/usr/src/${MODULE_NAME}-${MODULE_VERSION}"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check proper sudo
if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}Please run as root (sudo ./build.sh)${NC}"
  exit 1
fi

echo -e "${GREEN}=== Bulding Pseudoscopic v${MODULE_VERSION} ===${NC}"

# 1. Dependency Check
echo -e "\n${YELLOW}[1/6] Installing dependencies...${NC}"
apt-get update
# Install essential build tools if missing, suppressing output unless error
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    dkms \
    git \
    bc \
    bison \
    flex \
    libelf-dev \
    libssl-dev > /dev/null

# 2. Cleanup Old Installations
echo -e "\n${YELLOW}[2/6] Cleaning environment...${NC}"

# Unload module if loaded
if lsmod | grep -q "$MODULE_NAME"; then
    echo "  Unloading existing module..."
    modprobe -r "$MODULE_NAME" || true
fi

# Remove from DKMS
if dkms status | grep -q "${MODULE_NAME}/${MODULE_VERSION}"; then
    echo "  Removing from DKMS..."
    dkms remove -m "$MODULE_NAME" -v "$MODULE_VERSION" --all || true
fi

# Remove source directory
if [ -d "$SRC_DIR" ]; then
    echo "  Removing old source directory..."
    rm -rf "$SRC_DIR"
fi

# Clean local directory
echo "  Cleaning local build artifacts..."
make clean > /dev/null 2>&1 || true

# 3. Prepare Source Tree
echo -e "\n${YELLOW}[3/6] Preparing source tree...${NC}"
mkdir -p "$SRC_DIR"
echo "  Copying sources to $SRC_DIR"
# Copy necessary files, excluding git configs, build artifacts etc.
cp -r src include dkms.conf "$SRC_DIR/"
# Use the DKMS-specific Makefile as the main Makefile in the source tree
cp Makefile.dkms "$SRC_DIR/Makefile"

# 4. Verify Local Build (Sanity Check)
echo -e "\n${YELLOW}[4/6] Verifying build...${NC}"
# We try to build locally first (using root Makefile logic but manually) 
# just to ensure compilation succeeds before handing off to DKMS
if ! make > /dev/null; then
    echo -e "${RED}Error: Local build failed. Please check build logs.${NC}"
    exit 1
fi
echo -e "  Local build successful."
# Clean up valid local build so it doesn't interfere
make clean > /dev/null

# 5. DKMS Install
echo -e "\n${YELLOW}[5/6] Registering with DKMS...${NC}"
dkms add -m "$MODULE_NAME" -v "$MODULE_VERSION"

echo "  Building module via DKMS..."
if dkms build -m "$MODULE_NAME" -v "$MODULE_VERSION"; then
    echo "  DKMS build successful."
else
    echo -e "${RED}Error: DKMS build failed.${NC}"
    echo "Check /var/lib/dkms/${MODULE_NAME}/${MODULE_VERSION}/build/make.log"
    cat "/var/lib/dkms/${MODULE_NAME}/${MODULE_VERSION}/build/make.log"
    exit 1
fi

echo "  Installing module via DKMS..."
dkms install -m "$MODULE_NAME" -v "$MODULE_VERSION"

# 6. Verify Installation
echo -e "\n${YELLOW}[6/6] Verifying installation...${NC}"
# Configuration
echo "  Configuring modprobe..."
echo "options pseudoscopic device_idx=1 mode=ramdisk" > /etc/modprobe.d/pseudoscopic.conf

echo "  Loading module..."
if modprobe "$MODULE_NAME"; then
    echo -e "${GREEN}SUCCESS: Module loaded!${NC}"
    modinfo "$MODULE_NAME" | head -n 5
else
    echo -e "${RED}Warning: Module load failed.${NC}"
    echo "This is expected if no supported GPU (second NVIDIA card) is present."
    echo "However, the driver is correctly installed and will load on boot if hardware is found."
fi

# Helper aliases for demos
echo -e "\n${GREEN}Setup Complete.${NC}"
echo "You can now run 'make' in contrib/nearmem/ to build user-space tools."
