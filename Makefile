# SPDX-License-Identifier: GPL-2.0
#
# Pseudoscopic - GPU VRAM as System RAM
# Main Makefile
#
# Usage:
#   make              - Build kernel module
#   make install      - Install module and DKMS
#   make clean        - Clean build artifacts
#   make check        - Run static analysis
#   make DEBUG=1      - Build with debug symbols
#
# Copyright (C) 2025 Neural Splines LLC

# Module name
MODULE_NAME := pseudoscopic
MODULE_VERSION := 0.0.1

# Kernel build directory
KDIR ?= /lib/modules/$(shell uname -r)/build

# Installation paths
DKMS_DIR := /usr/src/$(MODULE_NAME)-$(MODULE_VERSION)
MODPROBE_DIR := /etc/modprobe.d
MODULES_LOAD_DIR := /etc/modules-load.d

#-----------------------------------------------------------------------------
# Targets
#-----------------------------------------------------------------------------

.PHONY: all modules clean install uninstall dkms-install dkms-remove check help

all: modules

# Build kernel module
modules:
	@echo "  Building kernel module..."
	$(MAKE) -C $(KDIR) M=$(PWD)/src modules
	# Copy the module to the root directory for convenience/scripts
	cp src/$(MODULE_NAME).ko .

# Clean all artifacts
clean:
	@echo "  Cleaning..."
	$(MAKE) -C $(KDIR) M=$(PWD)/src clean 2>/dev/null || true
	rm -f $(MODULE_NAME).ko
	rm -f Module.symvers modules.order
	rm -rf .tmp_versions

# Install module with DKMS
install: dkms-install
	@echo "  Installing modprobe configuration..."
	install -Dm644 dkms/pseudoscopic.conf $(MODPROBE_DIR)/pseudoscopic.conf
	@echo "  Enabling module load at boot..."
	echo "pseudoscopic" > $(MODULES_LOAD_DIR)/pseudoscopic.conf
	@echo ""
	@echo "  Installation complete!"
	@echo "  Load with: sudo modprobe pseudoscopic"
	@echo ""

# Uninstall module
uninstall: dkms-remove
	@echo "  Removing configuration files..."
	rm -f $(MODPROBE_DIR)/pseudoscopic.conf
	rm -f $(MODULES_LOAD_DIR)/pseudoscopic.conf
	@echo "  Uninstallation complete."

# DKMS installation
dkms-install:
	@echo "  Installing DKMS module..."
	@mkdir -p $(DKMS_DIR)
	cp -r src include Makefile.dkms dkms.conf $(DKMS_DIR)/
	mv $(DKMS_DIR)/Makefile.dkms $(DKMS_DIR)/Makefile
	dkms add -m $(MODULE_NAME) -v $(MODULE_VERSION)
	dkms build -m $(MODULE_NAME) -v $(MODULE_VERSION)
	dkms install -m $(MODULE_NAME) -v $(MODULE_VERSION)

# DKMS removal
dkms-remove:
	@echo "  Removing DKMS module..."
	-dkms remove -m $(MODULE_NAME) -v $(MODULE_VERSION) --all
	rm -rf $(DKMS_DIR)

# Static analysis
check:
	@echo "  Running sparse..."
	$(MAKE) -C $(KDIR) M=$(PWD)/src C=2 modules 2>&1 | grep -v "^make"
	@echo "  Running checkpatch..."
	$(KDIR)/scripts/checkpatch.pl --no-tree -f src/core/*.c src/hmm/*.c src/dma/*.c || true

# Generate compile_commands.json for IDE integration
compile_commands.json:
	@echo "  Generating compile_commands.json..."
	$(MAKE) -C $(KDIR) M=$(PWD)/src compile_commands.json

# Help target
help:
	@echo "Pseudoscopic - GPU VRAM as System RAM"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build kernel module (default)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install module via DKMS"
	@echo "  uninstall - Remove module and DKMS"
	@echo "  check     - Run static analysis"
	@echo "  help      - Show this message"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1   - Build with debug symbols"
	@echo "  KDIR=path - Use specific kernel headers"
	@echo ""
	@echo "Example:"
	@echo "  make DEBUG=1"
	@echo "  sudo make install"
	@echo "  sudo modprobe pseudoscopic"
