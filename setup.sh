#!/bin/bash
#
# Pseudoscopic Complete Installation Script
#
# This script installs:
#   1. Pseudoscopic kernel driver (GPU VRAM as block device)
#   2. Near-memory computing library
#   3. C++ wrapper and headers
#   4. Example programs and debugger
#
# Requirements:
#   - Linux kernel headers for your running kernel
#   - GCC matching kernel compiler version
#   - CUDA toolkit (optional, for GPU acceleration)
#   - Root access for driver installation
#
# Copyright (C) 2026 Neural Splines LLC
# License: MIT
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Banner
print_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║   ██████╗ ███████╗███████╗██╗   ██╗██████╗  ██████╗                  ║"
    echo "║   ██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔═══██╗                 ║"
    echo "║   ██████╔╝███████╗█████╗  ██║   ██║██║  ██║██║   ██║                 ║"
    echo "║   ██╔═══╝ ╚════██║██╔══╝  ██║   ██║██║  ██║██║   ██║                 ║"
    echo "║   ██║     ███████║███████╗╚██████╔╝██████╔╝╚██████╔╝                 ║"
    echo "║   ╚═╝     ╚══════╝╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝SCOPIC            ║"
    echo "║                                                                      ║"
    echo "║   Near-Memory Computing via GPU VRAM                                 ║"
    echo "║   Version 0.0.1                                                      ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Detect script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Installation prefix
PREFIX="${PREFIX:-/usr/local}"
KERNEL_VERSION=$(uname -r)

# Check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        log_error "This script must be run as root (for driver installation)"
        log_info "Try: sudo $0 $*"
        exit 1
    fi
}

# Detect kernel compiler version
detect_kernel_compiler() {
    log_info "Detecting kernel compiler version..." >&2
    
    # Method 1: Check /proc/version
    if [[ -f /proc/version ]]; then
        KERNEL_GCC_VERSION=$(cat /proc/version | grep -oP 'gcc[- ]version \K[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$KERNEL_GCC_VERSION" ]]; then
            log_info "Kernel compiled with GCC $KERNEL_GCC_VERSION (from /proc/version)" >&2
            echo "$KERNEL_GCC_VERSION"
            return 0
        fi
    fi
    
    # Method 2: Check kernel config
    KERNEL_CONFIG="/boot/config-$KERNEL_VERSION"
    if [[ -f "$KERNEL_CONFIG" ]]; then
        KERNEL_GCC_VERSION=$(grep CONFIG_CC_VERSION_TEXT "$KERNEL_CONFIG" 2>/dev/null | \
                            grep -oP 'gcc[- ]\([^)]+\) \K[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$KERNEL_GCC_VERSION" ]]; then
            log_info "Kernel compiled with GCC $KERNEL_GCC_VERSION (from config)" >&2
            echo "$KERNEL_GCC_VERSION"
            return 0
        fi
    fi
    
    # Method 3: Check vmlinux if available
    if [[ -f "/usr/lib/debug/boot/vmlinux-$KERNEL_VERSION" ]]; then
        KERNEL_GCC_VERSION=$(strings "/usr/lib/debug/boot/vmlinux-$KERNEL_VERSION" 2>/dev/null | \
                            grep -oP 'GCC: \([^)]+\) \K[0-9]+\.[0-9]+' | head -1)
        if [[ -n "$KERNEL_GCC_VERSION" ]]; then
            log_info "Kernel compiled with GCC $KERNEL_GCC_VERSION (from vmlinux)" >&2
            echo "$KERNEL_GCC_VERSION"
            return 0
        fi
    fi
    
    # Fallback: use current GCC
    log_warn "Could not detect kernel compiler version, using system GCC" >&2
    gcc --version | head -1 | grep -oP '\K[0-9]+\.[0-9]+' | head -1
}

# Find matching GCC
find_matching_gcc() {
    local target_version="$1"
    local major_version="${target_version%%.*}"
    
    log_info "Looking for GCC $target_version (or compatible)..." >&2
    
    # Try exact version first
    for gcc_path in /usr/bin/gcc-$target_version /usr/bin/gcc-$major_version gcc; do
        if command -v "$gcc_path" &>/dev/null; then
            local found_version=$($gcc_path --version | head -1 | grep -oP '\K[0-9]+\.[0-9]+' | head -1)
            local found_major="${found_version%%.*}"
            
            if [[ "$found_major" == "$major_version" ]]; then
                log_success "Found compatible GCC: $gcc_path (version $found_version)" >&2
                echo "$gcc_path"
                return 0
            fi
        fi
    done
    
    # Check if default gcc is close enough
    local system_version=$(gcc --version | head -1 | grep -oP '\K[0-9]+\.[0-9]+' | head -1)
    local system_major="${system_version%%.*}"
    
    if [[ "$system_major" == "$major_version" ]]; then
        log_success "System GCC is compatible (version $system_version)" >&2
        echo "gcc"
        return 0
    fi
    
    log_error "No compatible GCC found!" >&2
    log_info "Kernel was compiled with GCC $target_version" >&2
    log_info "Please install gcc-$major_version:" >&2
    log_info "  Ubuntu/Debian: apt install gcc-$major_version" >&2
    log_info "  Fedora/RHEL:   dnf install gcc" >&2
    return 1
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local missing=()
    
    # Kernel headers
    if [[ ! -d "/lib/modules/$KERNEL_VERSION/build" ]]; then
        missing+=("linux-headers-$KERNEL_VERSION")
    else
        log_success "Kernel headers found"
    fi
    
    # Build tools
    for tool in make gcc g++; do
        if ! command -v $tool &>/dev/null; then
            missing+=("$tool")
        else
            log_success "$tool found: $(command -v $tool)"
        fi
    done
    
    # DKMS (optional but recommended)
    if ! command -v dkms &>/dev/null; then
        log_warn "DKMS not found (recommended for automatic driver rebuild)"
    else
        log_success "DKMS found"
    fi
    
    # CUDA (optional)
    if [[ -d "/usr/local/cuda" ]] || command -v nvcc &>/dev/null; then
        CUDA_PATH="${CUDA_PATH:-/usr/local/cuda}"
        log_success "CUDA found: $CUDA_PATH"
        HAVE_CUDA=1
    else
        log_warn "CUDA not found (GPU acceleration disabled)"
        HAVE_CUDA=0
    fi
    
    if [[ ${#missing[@]} -gt 0 ]]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Install with:"
        log_info "  Ubuntu/Debian: apt install ${missing[*]}"
        log_info "  Fedora/RHEL:   dnf install ${missing[*]}"
        return 1
    fi
    
    return 0
}

# Detect NVIDIA GPUs
detect_gpus() {
    log_info "Detecting NVIDIA GPUs..."
    
    GPU_COUNT=0
    GPU_INFO=()
    
    # Use lspci to find NVIDIA GPUs
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            GPU_INFO+=("$line")
            ((GPU_COUNT++))
        fi
    done < <(lspci -nn | grep -i 'nvidia' | grep -iE 'vga|3d|display')
    
    if [[ $GPU_COUNT -eq 0 ]]; then
        log_warn "No NVIDIA GPUs detected"
        log_info "Pseudoscopic requires NVIDIA GPUs with BAR1 access"
        return 1
    fi
    
    log_success "Found $GPU_COUNT NVIDIA GPU(s):"
    for i in "${!GPU_INFO[@]}"; do
        log_info "  GPU $i: ${GPU_INFO[$i]}"
    done
    
    return 0
}

# Build and install kernel driver
install_driver() {
    log_info "Building pseudoscopic kernel driver..."
    
    local driver_dir="$PROJECT_ROOT"
    
    if [[ ! -d "$driver_dir" ]]; then
        log_error "Driver source not found at $driver_dir"
        return 1
    fi
    
    # Get matching GCC
    local kernel_gcc_version=$(detect_kernel_compiler)
    local gcc_path=$(find_matching_gcc "$kernel_gcc_version")
    
    if [[ -z "$gcc_path" ]]; then
        return 1
    fi
    
    # Build driver
    cd "$driver_dir"
    
    log_info "Building with CC=$gcc_path..."
    make clean 2>/dev/null || true
    make CC="$gcc_path"
    
    if [[ ! -f "pseudoscopic.ko" ]]; then
        log_error "Driver build failed"
        return 1
    fi
    
    log_success "Driver built successfully"
    
    # Install driver
    log_info "Installing driver module..."
    
    local module_dir="/lib/modules/$KERNEL_VERSION/extra"
    mkdir -p "$module_dir"
    cp pseudoscopic.ko "$module_dir/"
    
    # Update module dependencies
    depmod -a
    
    # Create udev rules
    log_info "Installing udev rules..."
    cat > /etc/udev/rules.d/99-pseudoscopic.rules << 'EOF'
# Pseudoscopic GPU VRAM block devices
KERNEL=="psdisk[0-9]*", SUBSYSTEM=="block", MODE="0666", GROUP="disk"
KERNEL=="psdisk[0-9]*", SUBSYSTEM=="block", TAG+="systemd"
EOF
    
    udevadm control --reload-rules
    
    # Load module
    log_info "Loading pseudoscopic module..."
    modprobe -r pseudoscopic 2>/dev/null || true
    modprobe pseudoscopic
    
    if lsmod | grep -q pseudoscopic; then
        log_success "Driver loaded successfully"
        
        # Check for devices
        sleep 1
        if ls /dev/psdisk* &>/dev/null; then
            log_success "Block devices created:"
            ls -la /dev/psdisk* 2>/dev/null | while read line; do
                log_info "  $line"
            done
        fi
    else
        log_error "Failed to load driver"
        return 1
    fi
    
    # Set up DKMS if available
    if command -v dkms &>/dev/null; then
        log_info "Setting up DKMS for automatic rebuilds..."
        
        # Use Makefile target for consistent DKMS setup
        if make dkms-install; then
            log_success "DKMS configured"
        else
            log_warn "DKMS configuration failed, but manual driver installation succeeded"
        fi
    fi
    
    # Enable at boot
    log_info "Enabling driver at boot..."
    echo "pseudoscopic" > /etc/modules-load.d/pseudoscopic.conf
    
    cd "$PROJECT_ROOT"
    return 0
}

# Build and install nearmem library
install_library() {
    log_info "Building near-memory library..."
    
    local lib_dir="$PROJECT_ROOT/contrib/nearmem"
    
    if [[ ! -d "$lib_dir" ]]; then
        log_error "Library source not found at $lib_dir"
        return 1
    fi
    
    cd "$lib_dir"
    
    # Clean and build
    make clean 2>/dev/null || true
    make lib
    
    if [[ ! -f "libnearmem.a" ]] || [[ ! -f "libnearmem.so" ]]; then
        log_error "Library build failed"
        return 1
    fi
    
    log_success "Library built successfully"
    
    # Install library
    log_info "Installing library to $PREFIX..."
    
    mkdir -p "$PREFIX/lib"
    mkdir -p "$PREFIX/include/nearmem"
    mkdir -p "$PREFIX/include/pseudoscopic"
    
    # Libraries
    cp libnearmem.a "$PREFIX/lib/"
    cp libnearmem.so "$PREFIX/lib/"
    
    # C headers
    cp include/*.h "$PREFIX/include/nearmem/"
    
    # C++ headers (if built)
    if [[ -f "include/nearmem.hpp" ]]; then
        cp include/nearmem.hpp "$PREFIX/include/nearmem/"
    fi
    
    # Driver headers
    cp "$PROJECT_ROOT/include/pseudoscopic/"*.h "$PREFIX/include/pseudoscopic/" 2>/dev/null || true
    
    # Update library cache
    echo "$PREFIX/lib" > /etc/ld.so.conf.d/pseudoscopic.conf
    ldconfig
    
    log_success "Library installed"
    
    # Build examples
    log_info "Building examples..."
    make examples
    
    # Install examples
    mkdir -p "$PREFIX/share/pseudoscopic/examples"
    for example in log_analyzer kv_cache_tier tiled_convolution tiled_matmul tiletrace gpucpu_demo gpufpga_demo abyssal_demo; do
        if [[ -f "$example" ]]; then
            cp "$example" "$PREFIX/share/pseudoscopic/examples/"
        fi
    done
    
    log_success "Examples installed to $PREFIX/share/pseudoscopic/examples/"
    
    cd "$PROJECT_ROOT"
    return 0
}

# Install pkg-config file
install_pkgconfig() {
    log_info "Installing pkg-config file..."
    
    mkdir -p "$PREFIX/lib/pkgconfig"
    
    cat > "$PREFIX/lib/pkgconfig/nearmem.pc" << EOF
prefix=$PREFIX
exec_prefix=\${prefix}
libdir=\${prefix}/lib
includedir=\${prefix}/include

Name: nearmem
Description: Near-Memory Computing Library for GPU VRAM
Version: 0.0.1
Requires:
Libs: -L\${libdir} -lnearmem -lpthread -ldl
Cflags: -I\${includedir}
EOF
    
    log_success "pkg-config file installed"
}

# Create system info command
install_info_tool() {
    log_info "Installing pseudoscopic-info tool..."
    
    cat > "$PREFIX/bin/pseudoscopic-info" << 'INFOEOF'
#!/bin/bash
#
# pseudoscopic-info - Display Pseudoscopic system information
#

echo "Pseudoscopic System Information"
echo "================================"
echo ""

# Driver status
echo "Driver Status:"
if lsmod | grep -q pseudoscopic; then
    echo "  Module:  LOADED"
    modinfo pseudoscopic 2>/dev/null | grep -E "^(version|description):" | sed 's/^/  /'
else
    echo "  Module:  NOT LOADED"
fi
echo ""

# Devices
echo "Block Devices:"
if ls /dev/psdisk* &>/dev/null; then
    for dev in /dev/psdisk*; do
        size=$(blockdev --getsize64 "$dev" 2>/dev/null || echo "unknown")
        if [[ "$size" != "unknown" ]]; then
            size_mb=$((size / 1024 / 1024))
            echo "  $dev: ${size_mb} MB"
        else
            echo "  $dev: size unknown"
        fi
    done
else
    echo "  No devices found"
fi
echo ""

# GPU Info
echo "NVIDIA GPUs:"
lspci -nn | grep -i nvidia | grep -iE 'vga|3d|display' | while read line; do
    echo "  $line"
done
echo ""

# Library
echo "Library:"
if pkg-config --exists nearmem 2>/dev/null; then
    echo "  Version: $(pkg-config --modversion nearmem)"
    echo "  Libs:    $(pkg-config --libs nearmem)"
else
    echo "  Not installed or pkg-config not configured"
fi
INFOEOF
    
    chmod +x "$PREFIX/bin/pseudoscopic-info"
    
    log_success "Info tool installed"
}

# Verify installation
verify_installation() {
    log_info "Verifying installation..."
    
    local errors=0
    
    # Check driver
    if ! lsmod | grep -q pseudoscopic; then
        log_error "Driver not loaded"
        ((errors++))
    fi
    
    # Check devices
    if ! ls /dev/psdisk* &>/dev/null; then
        log_warn "No block devices found (may be normal if no compatible GPU)"
    fi
    
    # Check library
    if [[ ! -f "$PREFIX/lib/libnearmem.so" ]]; then
        log_error "Shared library not installed"
        ((errors++))
    fi
    
    # Check headers
    if [[ ! -f "$PREFIX/include/nearmem/nearmem.h" ]]; then
        log_error "Headers not installed"
        ((errors++))
    fi
    
    # Check pkg-config
    if ! pkg-config --exists nearmem 2>/dev/null; then
        log_warn "pkg-config not working (may need to refresh LD cache)"
    fi
    
    if [[ $errors -eq 0 ]]; then
        log_success "Installation verified successfully"
        return 0
    else
        log_error "Installation verification failed with $errors errors"
        return 1
    fi
}

# Print usage information
print_usage() {
    log_info "Installation complete!"
    echo ""
    echo "Usage Examples:"
    echo ""
    echo "  # Check system status"
    echo "  pseudoscopic-info"
    echo ""
    echo "  # Compile a program"
    echo "  gcc myprogram.c \$(pkg-config --cflags --libs nearmem) -o myprogram"
    echo ""
    echo "  # Run examples"
    echo "  cd $PREFIX/share/pseudoscopic/examples"
    echo "  ./abyssal_demo"
    echo ""
    echo "  # Use C++ wrapper"
    echo "  g++ -std=c++17 myprogram.cpp \$(pkg-config --cflags --libs nearmem) -o myprogram"
    echo ""
}

# Main installation
main() {
    print_banner
    
    # Parse arguments
    case "${1:-}" in
        --help|-h)
            echo "Usage: $0 [--driver-only|--lib-only|--uninstall]"
            echo ""
            echo "Options:"
            echo "  --driver-only   Install only the kernel driver"
            echo "  --lib-only      Install only the library (skip driver)"
            echo "  --uninstall     Remove all installed components"
            echo "  --help          Show this help"
            exit 0
            ;;
        --uninstall)
            check_root
            log_info "Uninstalling Pseudoscopic..."
            
            # Unload driver
            modprobe -r pseudoscopic 2>/dev/null || true
            
            # Remove DKMS
            dkms remove -m pseudoscopic -v 0.0.1 --all 2>/dev/null || true
            rm -rf /usr/src/pseudoscopic-0.0.1
            
            # Remove driver
            rm -f /lib/modules/*/extra/pseudoscopic.ko
            rm -f /etc/modules-load.d/pseudoscopic.conf
            rm -f /etc/udev/rules.d/99-pseudoscopic.rules
            depmod -a
            
            # Remove library
            rm -f "$PREFIX/lib/libnearmem.a"
            rm -f "$PREFIX/lib/libnearmem.so"
            rm -rf "$PREFIX/include/nearmem"
            rm -rf "$PREFIX/include/pseudoscopic"
            rm -f "$PREFIX/lib/pkgconfig/nearmem.pc"
            rm -rf "$PREFIX/share/pseudoscopic"
            rm -f "$PREFIX/bin/pseudoscopic-info"
            rm -f /etc/ld.so.conf.d/pseudoscopic.conf
            ldconfig
            
            log_success "Uninstallation complete"
            exit 0
            ;;
    esac
    
    check_root
    
    log_info "Starting installation..."
    log_info "Prefix: $PREFIX"
    log_info "Kernel: $KERNEL_VERSION"
    
    # Check dependencies
    if ! check_dependencies; then
        exit 1
    fi
    
    # Detect GPUs
    detect_gpus || true  # Continue even without GPU
    
    # Install driver (unless --lib-only)
    if [[ "${1:-}" != "--lib-only" ]]; then
        if ! install_driver; then
            log_error "Driver installation failed"
            exit 1
        fi
    fi
    
    # Install library (unless --driver-only)
    if [[ "${1:-}" != "--driver-only" ]]; then
        if ! install_library; then
            log_error "Library installation failed"
            exit 1
        fi
        
        install_pkgconfig
        install_info_tool
    fi
    
    # Verify
    if verify_installation; then
        print_usage
    fi
}

main "$@"
