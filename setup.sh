#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# setup.sh - Bulletproof Environment Setup for Pseudoscopic
#
# This script prepares your system for building and running Pseudoscopic.
# After running this script, you can simply run `make` to build the kernel module.
#
# Usage:
#   ./setup.sh              # Interactive mode (will prompt for sudo)
#   sudo ./setup.sh         # Run with root privileges directly
#   ./setup.sh --check      # Check dependencies without installing
#   ./setup.sh --cuda       # Also install CUDA toolkit for nearmem library
#   ./setup.sh --full       # Full install: all deps + CUDA + nearmem library
#
# Copyright (C) 2025 Neural Splines LLC
# Author: Robert L. Sitton, Jr.

set -e

#-----------------------------------------------------------------------------
# Configuration
#-----------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Supported distros
SUPPORTED_DISTROS="Ubuntu 22.04+, Debian 12+, Fedora 38+, Arch Linux"

# Minimum kernel version for HMM support
MIN_KERNEL_MAJOR=6
MIN_KERNEL_MINOR=5

#-----------------------------------------------------------------------------
# Utility Functions
#-----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo ""
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}${CYAN}  $1${NC}"
    echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

command_exists() {
    command -v "$1" &> /dev/null
}

require_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This operation requires root privileges"
        log_info "Run: sudo $SCRIPT_DIR/$SCRIPT_NAME $*"
        exit 1
    fi
}

detect_distro() {
    if [[ -f /etc/os-release ]]; then
        . /etc/os-release
        DISTRO_ID="$ID"
        DISTRO_VERSION="$VERSION_ID"
        DISTRO_NAME="$NAME"
    elif command_exists lsb_release; then
        DISTRO_ID=$(lsb_release -si | tr '[:upper:]' '[:lower:]')
        DISTRO_VERSION=$(lsb_release -sr)
        DISTRO_NAME=$(lsb_release -sd)
    else
        DISTRO_ID="unknown"
        DISTRO_VERSION="unknown"
        DISTRO_NAME="Unknown Distribution"
    fi
}

get_package_manager() {
    if command_exists apt-get; then
        PKG_MANAGER="apt"
        PKG_UPDATE="apt-get update"
        PKG_INSTALL="apt-get install -y"
    elif command_exists dnf; then
        PKG_MANAGER="dnf"
        PKG_UPDATE="dnf check-update || true"
        PKG_INSTALL="dnf install -y"
    elif command_exists yum; then
        PKG_MANAGER="yum"
        PKG_UPDATE="yum check-update || true"
        PKG_INSTALL="yum install -y"
    elif command_exists pacman; then
        PKG_MANAGER="pacman"
        PKG_UPDATE="pacman -Sy"
        PKG_INSTALL="pacman -S --noconfirm"
    elif command_exists zypper; then
        PKG_MANAGER="zypper"
        PKG_UPDATE="zypper refresh"
        PKG_INSTALL="zypper install -y"
    else
        PKG_MANAGER="unknown"
        log_error "Could not detect package manager"
        return 1
    fi
}

#-----------------------------------------------------------------------------
# Dependency Checking
#-----------------------------------------------------------------------------

check_kernel_version() {
    local kernel_version
    kernel_version=$(uname -r)
    local major minor
    major=$(echo "$kernel_version" | cut -d. -f1)
    minor=$(echo "$kernel_version" | cut -d. -f2)
    
    if [[ $major -lt $MIN_KERNEL_MAJOR ]] || \
       [[ $major -eq $MIN_KERNEL_MAJOR && $minor -lt $MIN_KERNEL_MINOR ]]; then
        log_error "Kernel version $kernel_version is too old"
        log_error "Pseudoscopic requires kernel $MIN_KERNEL_MAJOR.$MIN_KERNEL_MINOR or newer for HMM support"
        return 1
    fi
    
    log_success "Kernel version: $kernel_version (>= $MIN_KERNEL_MAJOR.$MIN_KERNEL_MINOR required)"
    return 0
}

check_kernel_headers() {
    local kernel_version
    kernel_version=$(uname -r)
    local headers_dir="/lib/modules/${kernel_version}/build"
    
    if [[ ! -d "$headers_dir" ]]; then
        log_warn "Kernel headers not found at: $headers_dir"
        return 1
    fi
    
    log_success "Kernel headers: $headers_dir"
    return 0
}

check_nasm() {
    if ! command_exists nasm; then
        log_warn "NASM assembler not found"
        return 1
    fi
    
    local version
    version=$(nasm -v 2>&1 | head -n1 | grep -oP '\d+\.\d+(\.\d+)?' || echo "unknown")
    log_success "NASM: $version"
    return 0
}

check_gcc() {
    if ! command_exists gcc; then
        log_warn "GCC not found"
        return 1
    fi
    
    local version
    version=$(gcc --version | head -n1 | grep -oP '\d+\.\d+\.\d+' | head -n1)
    version=${version:-unknown}
    log_success "GCC: $version"
    return 0
}

check_make() {
    if ! command_exists make; then
        log_warn "Make not found"
        return 1
    fi
    
    local version
    version=$(make --version | head -n1 | grep -oP '\d+\.\d+' || echo "unknown")
    log_success "Make: $version"
    return 0
}

check_dkms() {
    if ! command_exists dkms; then
        log_warn "DKMS not found (optional, for automatic kernel updates)"
        return 1
    fi
    
    local version
    version=$(dkms --version 2>&1 | grep -oP '\d+\.\d+(\.\d+)?' || echo "unknown")
    log_success "DKMS: $version"
    return 0
}

check_nvidia_gpu() {
    if ! lspci 2>/dev/null | grep -qi "nvidia"; then
        log_warn "No NVIDIA GPU detected in lspci output"
        return 1
    fi
    
    local gpu_info
    gpu_info=$(lspci 2>/dev/null | grep -i nvidia | head -n1 | cut -d: -f3 | xargs)
    log_success "NVIDIA GPU: $gpu_info"
    return 0
}

check_cuda() {
    local cuda_path="${CUDA_PATH:-/usr/local/cuda}"
    
    if [[ ! -d "$cuda_path" ]]; then
        # Check alternative locations
        for path in /usr/local/cuda-* /opt/cuda*; do
            if [[ -d "$path" ]]; then
                cuda_path="$path"
                break
            fi
        done
    fi
    
    if [[ ! -x "$cuda_path/bin/nvcc" ]]; then
        log_warn "CUDA toolkit not found (optional, for nearmem library GPU acceleration)"
        return 1
    fi
    
    local version
    version=$("$cuda_path/bin/nvcc" --version 2>&1 | grep -oP 'release \K\d+\.\d+' || echo "unknown")
    log_success "CUDA: $version at $cuda_path"
    CUDA_DETECTED="$cuda_path"
    return 0
}

check_nouveau() {
    if lsmod 2>/dev/null | grep -q "^nouveau"; then
        log_warn "nouveau driver is loaded (may conflict with pseudoscopic)"
        NOUVEAU_LOADED=true
        return 1
    fi
    
    log_success "nouveau driver: not loaded"
    NOUVEAU_LOADED=false
    return 0
}

check_nvidia_driver() {
    if lsmod 2>/dev/null | grep -q "^nvidia "; then
        log_warn "nvidia proprietary driver is loaded"
        log_info "  For ramdisk mode: nvidia driver can coexist"
        log_info "  For RAM mode: may need to unload nvidia driver"
        NVIDIA_LOADED=true
        return 0
    fi
    
    NVIDIA_LOADED=false
    return 0
}

#-----------------------------------------------------------------------------
# Package Installation
#-----------------------------------------------------------------------------

install_base_deps() {
    log_header "Installing Base Dependencies"
    
    require_root
    
    log_info "Updating package lists..."
    $PKG_UPDATE 2>/dev/null || true
    
    case "$PKG_MANAGER" in
        apt)
            local kernel_version
            kernel_version=$(uname -r)
            
            log_info "Installing build essentials..."
            $PKG_INSTALL \
                build-essential \
                linux-headers-"$kernel_version" \
                nasm \
                dkms \
                git \
                pkg-config \
                || { log_error "Failed to install packages"; return 1; }
            ;;
        
        dnf|yum)
            local kernel_version
            kernel_version=$(uname -r)
            
            log_info "Installing development tools..."
            $PKG_INSTALL \
                kernel-devel-"$kernel_version" \
                kernel-headers-"$kernel_version" \
                gcc \
                make \
                nasm \
                dkms \
                git \
                pkgconfig \
                || { log_error "Failed to install packages"; return 1; }
            ;;
        
        pacman)
            log_info "Installing development tools..."
            $PKG_INSTALL \
                base-devel \
                linux-headers \
                nasm \
                dkms \
                git \
                || { log_error "Failed to install packages"; return 1; }
            ;;
        
        zypper)
            log_info "Installing development tools..."
            $PKG_INSTALL \
                kernel-devel \
                gcc \
                make \
                nasm \
                dkms \
                git \
                || { log_error "Failed to install packages"; return 1; }
            ;;
        
        *)
            log_error "Unsupported package manager: $PKG_MANAGER"
            log_info "Please install manually:"
            log_info "  - GCC (build-essential)"
            log_info "  - Kernel headers for $(uname -r)"
            log_info "  - NASM assembler"
            log_info "  - DKMS (optional)"
            return 1
            ;;
    esac
    
    log_success "Base dependencies installed successfully"
    return 0
}

install_cuda_toolkit() {
    log_header "Installing CUDA Toolkit"
    
    require_root
    
    log_info "CUDA installation varies by distribution and GPU."
    log_info "This will attempt to install CUDA from NVIDIA's repository."
    echo ""
    
    case "$PKG_MANAGER" in
        apt)
            # Ubuntu/Debian CUDA installation
            log_info "Adding NVIDIA CUDA repository..."
            
            # Install keyring
            if [[ ! -f /usr/share/keyrings/cuda-archive-keyring.gpg ]]; then
                local dist_codename
                dist_codename=$(lsb_release -cs 2>/dev/null || echo "jammy")
                
                # For Ubuntu versions not yet supported, fall back to latest
                case "$dist_codename" in
                    noble|mantic|lunar)
                        dist_codename="jammy"  # Fall back to 22.04
                        ;;
                esac
                
                wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb" \
                    -O /tmp/cuda-keyring.deb 2>/dev/null || {
                    log_error "Failed to download CUDA keyring"
                    log_info "Manual install: https://developer.nvidia.com/cuda-downloads"
                    return 1
                }
                dpkg -i /tmp/cuda-keyring.deb
                rm -f /tmp/cuda-keyring.deb
            fi
            
            $PKG_UPDATE
            $PKG_INSTALL cuda-toolkit-12-4 || {
                log_warn "CUDA 12.4 not available, trying generic cuda-toolkit..."
                $PKG_INSTALL cuda-toolkit || {
                    log_error "Failed to install CUDA toolkit"
                    return 1
                }
            }
            ;;
        
        dnf|yum)
            # Fedora/RHEL CUDA installation
            log_info "Adding NVIDIA CUDA repository..."
            
            if [[ ! -f /etc/yum.repos.d/cuda-rhel*.repo ]]; then
                $PKG_INSTALL dnf-plugins-core
                dnf config-manager --add-repo \
                    https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
            fi
            
            $PKG_INSTALL cuda-toolkit || {
                log_error "Failed to install CUDA toolkit"
                return 1
            }
            ;;
        
        pacman)
            # Arch Linux CUDA installation
            $PKG_INSTALL cuda || {
                log_error "Failed to install CUDA toolkit"
                return 1
            }
            ;;
        
        *)
            log_error "Automatic CUDA installation not supported for $PKG_MANAGER"
            log_info "Please install CUDA manually from:"
            log_info "  https://developer.nvidia.com/cuda-downloads"
            return 1
            ;;
    esac
    
    # Set up environment
    if [[ -d /usr/local/cuda ]]; then
        log_info "Setting up CUDA environment..."
        
        # Create profile.d script for persistent PATH
        cat > /etc/profile.d/cuda.sh << 'EOF'
# CUDA Toolkit environment
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=$CUDA_PATH/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
EOF
        chmod 644 /etc/profile.d/cuda.sh
        
        # Update ldconfig
        echo "/usr/local/cuda/lib64" > /etc/ld.so.conf.d/cuda.conf
        ldconfig 2>/dev/null || true
    fi
    
    log_success "CUDA toolkit installed successfully"
    log_info "Log out and back in, or run: source /etc/profile.d/cuda.sh"
    return 0
}

#-----------------------------------------------------------------------------
# Pre-flight Validation
#-----------------------------------------------------------------------------

run_checks() {
    log_header "Checking Dependencies"
    
    local all_pass=true
    local critical_fail=false
    
    # Critical checks (must pass)
    check_kernel_version || { critical_fail=true; all_pass=false; }
    check_kernel_headers || all_pass=false
    check_nasm || all_pass=false
    check_gcc || all_pass=false
    check_make || all_pass=false
    
    # Optional but recommended
    check_dkms || true
    
    # Hardware checks
    echo ""
    log_info "Hardware Detection:"
    check_nvidia_gpu || true
    check_nouveau || true
    check_nvidia_driver || true
    check_cuda || true
    
    echo ""
    
    if [[ "$critical_fail" == true ]]; then
        log_error "Critical requirements not met. Please upgrade your kernel."
        return 1
    fi
    
    if [[ "$all_pass" == true ]]; then
        log_success "All required dependencies are installed!"
        return 0
    else
        log_warn "Some dependencies are missing."
        return 1
    fi
}

#-----------------------------------------------------------------------------
# Build Test
#-----------------------------------------------------------------------------

test_build() {
    log_header "Testing Build"
    
    cd "$SCRIPT_DIR"
    
    log_info "Running: make clean"
    make clean 2>/dev/null || true
    
    log_info "Running: make"
    if make; then
        log_success "Kernel module built successfully!"
        log_info "Output: pseudoscopic.ko"
        
        if [[ -f "$SCRIPT_DIR/pseudoscopic.ko" ]]; then
            local size
            size=$(du -h "$SCRIPT_DIR/pseudoscopic.ko" | cut -f1)
            log_success "Module size: $size"
        fi
        
        return 0
    else
        log_error "Build failed. Check the output above for errors."
        return 1
    fi
}

build_nearmem() {
    log_header "Building nearmem Library"
    
    local nearmem_dir="$SCRIPT_DIR/contrib/nearmem"
    
    if [[ ! -d "$nearmem_dir" ]]; then
        log_warn "nearmem directory not found at: $nearmem_dir"
        return 1
    fi
    
    cd "$nearmem_dir"
    
    log_info "Running: make clean"
    make clean 2>/dev/null || true
    
    # Set CUDA_PATH if detected
    if [[ -n "$CUDA_DETECTED" ]]; then
        export CUDA_PATH="$CUDA_DETECTED"
        log_info "Using CUDA at: $CUDA_PATH"
    fi
    
    log_info "Running: make"
    if make; then
        log_success "nearmem library built successfully!"
        
        if [[ -f "$nearmem_dir/libnearmem.a" ]]; then
            log_success "Static library: libnearmem.a"
        fi
        if [[ -f "$nearmem_dir/libnearmem.so" ]]; then
            log_success "Shared library: libnearmem.so"
        fi
        if [[ -f "$nearmem_dir/log_analyzer" ]]; then
            log_success "Example: log_analyzer"
        fi
        
        return 0
    else
        log_error "nearmem build failed. Check the output above."
        return 1
    fi
}

#-----------------------------------------------------------------------------
# Post-Install Configuration
#-----------------------------------------------------------------------------

configure_nouveau_blacklist() {
    log_header "Configuring nouveau Blacklist"
    
    require_root
    
    local blacklist_file="/etc/modprobe.d/blacklist-nouveau.conf"
    
    if [[ -f "$blacklist_file" ]]; then
        log_info "nouveau blacklist already exists at: $blacklist_file"
        return 0
    fi
    
    log_info "Creating nouveau blacklist..."
    cat > "$blacklist_file" << 'EOF'
# Blacklist nouveau to prevent conflicts with pseudoscopic
# This is required for RAM mode, optional for ramdisk mode
blacklist nouveau
options nouveau modeset=0
EOF
    
    log_info "Regenerating initramfs..."
    if command_exists update-initramfs; then
        update-initramfs -u
    elif command_exists dracut; then
        dracut --force
    elif command_exists mkinitcpio; then
        mkinitcpio -P
    else
        log_warn "Could not regenerate initramfs automatically"
        log_info "Please regenerate manually for your distribution"
    fi
    
    log_success "nouveau blacklist configured"
    log_warn "Reboot required for blacklist to take effect"
    return 0
}

#-----------------------------------------------------------------------------
# Main Entry Points
#-----------------------------------------------------------------------------

show_usage() {
    echo "Pseudoscopic Setup Script"
    echo ""
    echo "Usage: $SCRIPT_NAME [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --check        Check dependencies without installing anything"
    echo "  --install      Install base dependencies (requires sudo)"
    echo "  --cuda         Install CUDA toolkit (requires sudo)"
    echo "  --nearmem      Build nearmem library"
    echo "  --full         Full setup: deps + CUDA + build + nearmem"
    echo "  --blacklist    Blacklist nouveau driver (requires sudo)"
    echo "  --help         Show this message"
    echo ""
    echo "Examples:"
    echo "  ./setup.sh                  # Interactive setup"
    echo "  ./setup.sh --check          # Just check dependencies"
    echo "  sudo ./setup.sh --install   # Install dependencies"
    echo "  ./setup.sh --full           # Complete setup"
    echo ""
    echo "Supported distributions: $SUPPORTED_DISTROS"
}

interactive_setup() {
    log_header "Pseudoscopic Setup"
    
    echo ""
    echo "  GPU VRAM as System RAM - Reversing depth perception"
    echo "  in the memory hierarchy."
    echo ""
    echo "  Copyright (C) 2025 Neural Splines LLC"
    echo ""
    
    # Detect environment
    detect_distro
    get_package_manager
    
    log_info "Detected: $DISTRO_NAME"
    log_info "Package manager: $PKG_MANAGER"
    
    # Run dependency checks
    if run_checks; then
        echo ""
        log_success "Environment is ready for building!"
        echo ""
        log_info "Next steps:"
        log_info "  1. Run: make"
        log_info "  2. Run: sudo make install"
        log_info "  3. Run: sudo modprobe pseudoscopic mode=ramdisk"
        log_info "  4. Verify: cat /proc/devices | grep psdisk"
        echo ""
        
        read -p "Would you like to build now? [Y/n] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Nn]$ ]]; then
            test_build
        fi
        
        return 0
    else
        echo ""
        log_warn "Some dependencies are missing."
        echo ""
        
        if [[ $EUID -eq 0 ]]; then
            read -p "Would you like to install missing dependencies? [Y/n] " -n 1 -r
            echo ""
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                install_base_deps
                echo ""
                run_checks
                echo ""
                test_build
            fi
        else
            log_info "To install dependencies, run:"
            log_info "  sudo $SCRIPT_DIR/$SCRIPT_NAME --install"
        fi
        
        return 1
    fi
}

full_setup() {
    log_header "Full Pseudoscopic Setup"
    
    detect_distro
    get_package_manager
    
    log_info "Detected: $DISTRO_NAME ($PKG_MANAGER)"
    
    # Check if we need sudo for installation
    if [[ $EUID -ne 0 ]]; then
        log_info "Requesting sudo for package installation..."
        sudo "$SCRIPT_DIR/$SCRIPT_NAME" --install-internal
    else
        install_base_deps
    fi
    
    # Re-check after installation
    run_checks || {
        log_error "Dependency installation may have failed"
        return 1
    }
    
    # Build kernel module
    test_build || {
        log_error "Kernel module build failed"
        return 1
    }
    
    # Check if CUDA is available, install if not
    if ! check_cuda; then
        echo ""
        read -p "Would you like to install CUDA toolkit for GPU acceleration? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            if [[ $EUID -ne 0 ]]; then
                sudo "$SCRIPT_DIR/$SCRIPT_NAME" --cuda-internal
            else
                install_cuda_toolkit
            fi
        fi
    fi
    
    # Build nearmem library
    build_nearmem || {
        log_warn "nearmem library build failed (optional)"
    }
    
    echo ""
    log_header "Setup Complete!"
    echo ""
    log_success "Pseudoscopic is ready to use!"
    echo ""
    log_info "Quick Start:"
    log_info "  sudo make install                              # Install via DKMS"
    log_info "  sudo modprobe pseudoscopic mode=ramdisk        # Load in ramdisk mode"
    log_info "  ls -la /dev/psdisk0                            # Verify device"
    log_info "  sudo mkfs.ext4 /dev/psdisk0                    # Format (optional)"
    log_info "  sudo mount /dev/psdisk0 /mnt/vram              # Mount"
    echo ""
    
    return 0
}

#-----------------------------------------------------------------------------
# Main
#-----------------------------------------------------------------------------

main() {
    # Parse command line arguments
    case "${1:-}" in
        --help|-h)
            show_usage
            exit 0
            ;;
        
        --check)
            detect_distro
            run_checks
            exit $?
            ;;
        
        --install)
            detect_distro
            get_package_manager
            install_base_deps
            run_checks
            exit $?
            ;;
        
        --install-internal)
            # Internal: called with sudo from full_setup
            detect_distro
            get_package_manager
            install_base_deps
            exit $?
            ;;
        
        --cuda)
            detect_distro
            get_package_manager
            install_cuda_toolkit
            exit $?
            ;;
        
        --cuda-internal)
            # Internal: called with sudo from full_setup
            detect_distro
            get_package_manager
            install_cuda_toolkit
            exit $?
            ;;
        
        --nearmem)
            detect_distro
            check_cuda || true
            build_nearmem
            exit $?
            ;;
        
        --full)
            full_setup
            exit $?
            ;;
        
        --blacklist)
            configure_nouveau_blacklist
            exit $?
            ;;
        
        --build)
            test_build
            exit $?
            ;;
        
        "")
            # No arguments: interactive mode
            interactive_setup
            exit $?
            ;;
        
        *)
            log_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
}

# Run main
main "$@"
