# SPDX-License-Identifier: MIT
#
# NMC - Near-Memory Computing Library
# Makefile
#
# Copyright (C) 2025 Neural Splines LLC

# Paths
CUDA_PATH ?= /usr/local/cuda
NMC_ROOT := $(shell pwd)
SRC_DIR := $(NMC_ROOT)/src
CUDA_DIR := $(NMC_ROOT)/cuda
INC_DIR := $(NMC_ROOT)/include
EXAMPLES_DIR := $(NMC_ROOT)/examples
BUILD_DIR := $(NMC_ROOT)/build
LIB_DIR := $(NMC_ROOT)/lib

# Compilers
CC := gcc
NVCC := $(CUDA_PATH)/bin/nvcc

# Flags
CFLAGS := -O3 -Wall -Wextra -fPIC -I$(INC_DIR) -I$(CUDA_PATH)/include
NVCCFLAGS := -O3 -Xcompiler -fPIC -I$(INC_DIR) --gpu-architecture=sm_60
LDFLAGS := -L$(CUDA_PATH)/lib64 -lcudart -lpthread

# Debug build
ifdef DEBUG
  CFLAGS += -g -O0 -DDEBUG
  NVCCFLAGS += -g -G -O0
endif

# Sources
NMC_SRCS := $(SRC_DIR)/nmc.c
CUDA_SRCS := $(CUDA_DIR)/nmc_kernels.cu
EXAMPLE_SRCS := $(wildcard $(EXAMPLES_DIR)/*.c)

# Objects
NMC_OBJS := $(BUILD_DIR)/nmc.o
CUDA_OBJS := $(BUILD_DIR)/nmc_kernels.o
EXAMPLE_BINS := $(patsubst $(EXAMPLES_DIR)/%.c,$(BUILD_DIR)/%,$(EXAMPLE_SRCS))

# Library
LIB_STATIC := $(LIB_DIR)/libnmc.a
LIB_SHARED := $(LIB_DIR)/libnmc.so

#-----------------------------------------------------------------------------
# Targets
#-----------------------------------------------------------------------------

.PHONY: all clean lib examples install

all: lib examples

# Create directories
$(BUILD_DIR) $(LIB_DIR):
	mkdir -p $@

# Compile C sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c | $(BUILD_DIR)
	@echo "  CC      $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA sources
$(BUILD_DIR)/%.o: $(CUDA_DIR)/%.cu | $(BUILD_DIR)
	@echo "  NVCC    $<"
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Static library
$(LIB_STATIC): $(NMC_OBJS) $(CUDA_OBJS) | $(LIB_DIR)
	@echo "  AR      $@"
	@ar rcs $@ $^

# Shared library
$(LIB_SHARED): $(NMC_OBJS) $(CUDA_OBJS) | $(LIB_DIR)
	@echo "  LD      $@"
	@$(CC) -shared $^ -o $@ $(LDFLAGS)

lib: $(LIB_STATIC) $(LIB_SHARED)

# Example programs
$(BUILD_DIR)/%: $(EXAMPLES_DIR)/%.c $(LIB_STATIC) | $(BUILD_DIR)
	@echo "  CC      $@"
	@$(CC) $(CFLAGS) $< -o $@ -L$(LIB_DIR) -lnmc $(LDFLAGS) -lm

examples: lib $(EXAMPLE_BINS)

# Install
PREFIX ?= /usr/local
install: lib
	@echo "Installing to $(PREFIX)..."
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIB_STATIC) $(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(PREFIX)/lib/
	install -m 644 $(INC_DIR)/nmc.h $(PREFIX)/include/
	ldconfig || true
	@echo "Done."

# Clean
clean:
	rm -rf $(BUILD_DIR) $(LIB_DIR)

# Help
help:
	@echo "NMC - Near-Memory Computing Library"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build library and examples (default)"
	@echo "  lib      - Build static and shared library"
	@echo "  examples - Build example programs"
	@echo "  install  - Install to PREFIX (default: /usr/local)"
	@echo "  clean    - Remove build artifacts"
	@echo ""
	@echo "Options:"
	@echo "  DEBUG=1    - Build with debug symbols"
	@echo "  CUDA_PATH  - Path to CUDA installation"
	@echo "  PREFIX     - Installation prefix"
	@echo ""
	@echo "Examples after build:"
	@echo "  ./build/log_grep /var/log/syslog \"error\""
	@echo "  ./build/analytics"
