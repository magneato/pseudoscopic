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

# Source directories
SRC_DIR := src
ASM_DIR := $(SRC_DIR)/asm

# Include directories
INCLUDES := -I$(PWD)/include

# C source files
C_SRCS := $(SRC_DIR)/core/module.c \
          $(SRC_DIR)/core/bar.c \
          $(SRC_DIR)/core/pool.c \
          $(SRC_DIR)/hmm/device.c \
          $(SRC_DIR)/hmm/migrate.c \
          $(SRC_DIR)/hmm/notifier.c \
          $(SRC_DIR)/dma/engine.c

# Assembly source files
ASM_SRCS := $(ASM_DIR)/memcpy_wc.asm \
            $(ASM_DIR)/cache_ops.asm \
            $(ASM_DIR)/barriers.asm

# Object files
C_OBJS := $(C_SRCS:.c=.o)
ASM_OBJS := $(ASM_SRCS:.asm=.o)
ALL_OBJS := $(C_OBJS) $(ASM_OBJS)

# NASM flags
NASMFLAGS := -f elf64 -g -F dwarf

# Debug build
ifdef DEBUG
  EXTRA_CFLAGS += -DDEBUG -g -O0
  NASMFLAGS += -g -F dwarf
else
  EXTRA_CFLAGS += -O2
endif

# Kernel module build flags
ccflags-y := $(INCLUDES)
ccflags-y += -Wall -Wextra -Werror
ccflags-y += -Wno-unused-parameter  # Kernel callbacks often have unused params

# Extra flags from command line
ccflags-y += $(EXTRA_CFLAGS)

# Module object composition
obj-m := $(MODULE_NAME).o
$(MODULE_NAME)-y := $(patsubst %.c,%.o,$(patsubst $(SRC_DIR)/%,%,$(C_SRCS)))
$(MODULE_NAME)-y += $(patsubst %.asm,%.o,$(patsubst $(SRC_DIR)/%,%,$(ASM_SRCS)))

# Installation paths
DKMS_DIR := /usr/src/$(MODULE_NAME)-$(MODULE_VERSION)
MODPROBE_DIR := /etc/modprobe.d
MODULES_LOAD_DIR := /etc/modules-load.d

#-----------------------------------------------------------------------------
# Targets
#-----------------------------------------------------------------------------

.PHONY: all modules clean install uninstall dkms-install dkms-remove check help

all: asm modules

# Build assembly files first
asm: $(ASM_OBJS)

# Assemble .asm files to .o
$(ASM_DIR)/%.o: $(ASM_DIR)/%.asm
	@echo "  NASM    $<"
	@nasm $(NASMFLAGS) -o $@ $<

# Build kernel module
modules: asm
	@echo "  Building kernel module..."
	$(MAKE) -C $(KDIR) M=$(PWD) modules

# Clean all artifacts
clean:
	@echo "  Cleaning..."
	$(MAKE) -C $(KDIR) M=$(PWD) clean 2>/dev/null || true
	rm -f $(ASM_DIR)/*.o
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
	$(MAKE) -C $(KDIR) M=$(PWD) C=2 modules 2>&1 | grep -v "^make"
	@echo "  Running checkpatch..."
	$(KDIR)/scripts/checkpatch.pl --no-tree -f $(C_SRCS) || true

# Generate compile_commands.json for IDE integration
compile_commands.json:
	@echo "  Generating compile_commands.json..."
	$(MAKE) -C $(KDIR) M=$(PWD) compile_commands.json

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
