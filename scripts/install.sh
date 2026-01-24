#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# install.sh - Pseudoscopic installation script
#
# Handles dependency checking, building, and installation
# with appropriate error handling.
#
# Copyright (C) 2025 Neural Splines LLC

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

echo -e "${BOLD}${CYAN}"
echo "╔═══════════════════════════════════════════════════════════════════╗"
echo "║                   Pseudoscopic Installer                          ║"
echo "║              GPU VRAM as System RAM for Linux                     ║"
echo "╚═══════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check root
if [[ $EUID -ne 0 ]]; then
    echo -e "${RED}Error: This script must be run as root${NC}"
    echo "Try: sudo $0"
    exit 1
fi

# Detect distribution
detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        echo "$ID"
    else
        echo "unknown"
    fi
}

DISTRO=$(detect_distro)
echo -e "Detected distribution: ${CYAN}${DISTRO}${NC}"

# Install dependencies
install_deps() {
    echo -e "\n${BOLD}Installing dependencies...${NC}"
    
    case "$DISTRO" in
        ubuntu|debian)
            apt-get update
            apt-get install -y \
                build-essential \
                linux-headers-$(uname -r) \
                dkms \
                nasm \
                bc
            ;;
        fedora)
            dnf install -y \
                kernel-devel \
                kernel-headers \
                dkms \
                nasm \
                bc \
                gcc \
                make
            ;;
        arch|manjaro)
            pacman -Sy --noconfirm \
                linux-headers \
                dkms \
                nasm \
                bc \
                base-devel
            ;;
        *)
            echo -e "${YELLOW}Warning: Unknown distribution. Please install manually:${NC}"
            echo "  - Kernel headers for $(uname -r)"
            echo "  - build-essential / base-devel"
            echo "  - DKMS"
            echo "  - NASM assembler"
            echo "  - bc calculator"
            read -p "Continue anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            ;;
    esac
    
    echo -e "${GREEN}Dependencies installed.${NC}"
}

# Check for NVIDIA GPU
check_gpu() {
    echo -e "\n${BOLD}Checking for compatible GPU...${NC}"
    
    if ! lspci | grep -qi nvidia; then
        echo -e "${RED}Error: No NVIDIA GPU detected${NC}"
        exit 1
    fi
    
    # Check for supported datacenter GPUs
    local gpu_info
    gpu_info=$(lspci -nn | grep -i nvidia | head -1)
    echo -e "  Found: ${CYAN}${gpu_info}${NC}"
    
    # Warn about consumer GPUs
    local supported_ids="15f8|15f9|1b38|1db4|1db1|1df6|1e30|20f1|20b0"
    if ! lspci -n | grep -qE "10de:(${supported_ids})"; then
        echo -e "${YELLOW}Warning: This doesn't appear to be a supported datacenter GPU.${NC}"
        echo -e "${YELLOW}Supported: Tesla P100, P40, V100, Quadro RTX, A100${NC}"
        echo -e "${YELLOW}Consumer GPUs may have limited BAR size.${NC}"
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo -e "${GREEN}Supported datacenter GPU detected!${NC}"
    fi
}

# Check for conflicting drivers
check_conflicts() {
    echo -e "\n${BOLD}Checking for driver conflicts...${NC}"
    
    if lsmod | grep -q "^nouveau"; then
        echo -e "${YELLOW}Warning: nouveau driver is loaded.${NC}"
        echo "Pseudoscopic requires direct hardware access."
        read -p "Unload nouveau? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            echo -e "${YELLOW}Skipping. You may need to blacklist nouveau manually.${NC}"
        else
            echo "Unloading nouveau..."
            modprobe -r nouveau || true
            echo "options nouveau modeset=0" > /etc/modprobe.d/blacklist-nouveau.conf
            echo "blacklist nouveau" >> /etc/modprobe.d/blacklist-nouveau.conf
            echo -e "${GREEN}nouveau blacklisted and unloaded.${NC}"
        fi
    else
        echo -e "${GREEN}No conflicting drivers found.${NC}"
    fi
}

# Build module
build_module() {
    echo -e "\n${BOLD}Building kernel module...${NC}"
    
    make clean 2>/dev/null || true
    
    if make; then
        echo -e "${GREEN}Build successful!${NC}"
    else
        echo -e "${RED}Build failed. Check errors above.${NC}"
        exit 1
    fi
}

# Install with DKMS
install_dkms() {
    echo -e "\n${BOLD}Installing via DKMS...${NC}"
    
    if make install; then
        echo -e "${GREEN}DKMS installation successful!${NC}"
    else
        echo -e "${RED}DKMS installation failed.${NC}"
        exit 1
    fi
}

# Load module
load_module() {
    echo -e "\n${BOLD}Loading module...${NC}"
    
    if modprobe pseudoscopic; then
        echo -e "${GREEN}Module loaded successfully!${NC}"
    else
        echo -e "${RED}Failed to load module. Check dmesg for errors.${NC}"
        dmesg | tail -20
        exit 1
    fi
}

# Verify installation
verify() {
    echo -e "\n${BOLD}Verifying installation...${NC}"
    
    if lsmod | grep -q "^pseudoscopic"; then
        echo -e "  ${GREEN}✓${NC} Module loaded"
    else
        echo -e "  ${RED}✗${NC} Module not loaded"
        return 1
    fi
    
    local version
    version=$(modinfo -F version pseudoscopic 2>/dev/null || echo "unknown")
    echo -e "  ${GREEN}✓${NC} Version: ${version}"
    
    # Check for kernel messages
    if dmesg | tail -20 | grep -q "pseudoscopic.*ready"; then
        echo -e "  ${GREEN}✓${NC} Device initialized"
    fi
    
    echo ""
    echo -e "${GREEN}${BOLD}Installation complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Run 'ps-status' to view current state"
    echo "  2. Run 'ps-test' to validate functionality"
    echo "  3. Check 'cat /proc/meminfo | grep Device'"
    echo ""
}

# Main installation flow
main() {
    install_deps
    check_gpu
    check_conflicts
    build_module
    install_dkms
    load_module
    verify
}

main "$@"
