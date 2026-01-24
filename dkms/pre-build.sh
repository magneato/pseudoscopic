#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
#
# pre-build.sh - DKMS pre-build hook
#
# Compiles NASM assembly files before the kernel build system
# is invoked. Required because the kernel build system doesn't
# natively understand .asm files.
#
# Copyright (C) 2025 Neural Splines LLC

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/../src"

echo "Pseudoscopic: Compiling assembly files..."

# Check for NASM
if ! command -v nasm &> /dev/null; then
    echo "ERROR: NASM assembler not found. Install with:"
    echo "  sudo apt install nasm"
    exit 1
fi

# Compile each assembly file
for asm_file in "${SRC_DIR}/asm"/*.asm; do
    if [[ -f "$asm_file" ]]; then
        obj_file="${asm_file%.asm}.o"
        echo "  NASM    $(basename "$asm_file")"
        nasm -f elf64 -g -F dwarf -o "$obj_file" "$asm_file"
    fi
done

echo "Pseudoscopic: Assembly compilation complete."
