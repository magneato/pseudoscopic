#!/bin/bash
#
# ramdisk.sh - Enable pseudoscopic ramdisk mode for gpucpu testing
#
# Usage: ./ramdisk.sh
#
# This script will:
#   1. Unload the current pseudoscopic module
#   2. Reload it in ramdisk mode to create /dev/psdisk0
#   3. Run the gpucpu_demo
#
# If ramdisk mode fails (e.g., on GT 1030), it will try direct BAR1 access
# which requires root privileges.
#

set -e

echo "=== Pseudoscopic VRAM Testing Setup ==="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script needs root to access GPU VRAM."
    echo "Re-running with sudo..."
    exec sudo "$0" "$@"
fi

# Unload current module if loaded
if lsmod | grep -q pseudoscopic; then
    echo "Unloading current pseudoscopic module..."
    rmmod pseudoscopic 2>/dev/null || true
    sleep 1
fi

# Try to load in ramdisk mode
echo "Loading pseudoscopic in ramdisk mode..."
if modprobe pseudoscopic mode=ramdisk 2>/dev/null; then
    echo "Module loaded successfully"
    sleep 1
    
    # Check if device was created
    if ls /dev/psdisk* 2>/dev/null; then
        echo ""
        echo "Block device created! Running demo..."
        echo ""
    else
        echo "Warning: Block device not created, will try direct BAR1 access"
    fi
else
    echo "Warning: Could not load module in ramdisk mode"
    echo "Will try direct BAR1 access instead"
    
    # Reload in RAM mode as fallback
    modprobe pseudoscopic mode=ram 2>/dev/null || true
fi

# Run the demo
cd "$(dirname "$0")"
echo ""
echo "Running gpucpu_demo..."
echo ""
LD_LIBRARY_PATH=. ./gpucpu_demo

echo ""
echo "Done!"
