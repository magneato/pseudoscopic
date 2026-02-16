# Near-Memory Computing Library Makefile
#
# Builds the nearmem library and examples.
# Requires CUDA toolkit for GPU-accelerated operations.
#
# Copyright (C) 2026 Neural Splines LLC

#==============================================================================
# BULLETPROOF CUDA DETECTION
#==============================================================================
# Try multiple common CUDA installation paths
CUDA_SEARCH_PATHS := /usr/local/cuda /usr/local/cuda-12 /usr/local/cuda-11 \
                     /opt/cuda /usr/lib/cuda /usr/cuda

# Find CUDA path - check each path for nvcc
define find_cuda_path
$(firstword $(foreach p,$(CUDA_SEARCH_PATHS),$(if $(wildcard $(p)/bin/nvcc),$(p),)))
endef

# Allow override via environment, otherwise auto-detect
ifeq ($(origin CUDA_PATH),undefined)
  CUDA_PATH := $(call find_cuda_path)
endif

# NVCC compiler path
NVCC := $(CUDA_PATH)/bin/nvcc

# Verify CUDA is actually available (nvcc exists and is executable)
CUDA_AVAILABLE := $(wildcard $(NVCC))

# Find libcuda.so - it's often in system lib dirs, not CUDA toolkit
# NVIDIA driver installs libcuda.so to system paths
CUDA_LIB_PATHS := /usr/lib/x86_64-linux-gnu /usr/lib64 \
                  $(CUDA_PATH)/lib64 $(CUDA_PATH)/lib64/stubs \
                  /usr/local/cuda/lib64/stubs

# Find first path that contains libcuda.so
define find_cuda_lib
$(firstword $(foreach p,$(CUDA_LIB_PATHS),$(if $(wildcard $(p)/libcuda.so*),$(p),)))
endef

CUDA_LIB_DIR := $(call find_cuda_lib)

#==============================================================================
# COMPILER SETTINGS
#==============================================================================
CC := gcc
CFLAGS := -Wall -Wextra -O2 -fPIC -I./include
LDFLAGS := -ldl -lpthread

# CUDA settings (if both nvcc and libcuda are available)
ifneq ($(CUDA_AVAILABLE),)
  ifneq ($(CUDA_LIB_DIR),)
    CUDA_ENABLED := yes
    NVCCFLAGS := -O2 -Xcompiler -fPIC -I./include -Wno-deprecated-gpu-targets
    LDFLAGS += -L$(CUDA_LIB_DIR) -lcuda
    CFLAGS += -DNEARMEM_HAS_CUDA
    # Also add CUDA toolkit lib path for other CUDA libs (cudart)
    ifneq ($(CUDA_PATH),)
      LDFLAGS += -L$(CUDA_PATH)/lib64 -lcudart
    endif
  endif
endif

#==============================================================================
# SOURCE FILES
#==============================================================================
LIB_SRCS := src/nearmem.c src/nearmem_tile.c src/gpucpu.c src/gpufpga.c src/abyssal.c src/nearmem_gpu.c
CUDA_SRCS := src/nearmem_kernels.cu src/gpucpu_kernels.cu
EXAMPLE_SRCS := examples/log_analyzer.c examples/kv_cache_tier.c examples/tiled_convolution.c \
                examples/tiled_matmul.c examples/tiletrace.c examples/gpucpu_demo.c \
                examples/gpufpga_demo.c examples/abyssal_demo.c

# Object files
LIB_OBJS := $(LIB_SRCS:.c=.o)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
EXAMPLE_OBJS := $(EXAMPLE_SRCS:.c=.o)

# Output
LIB_STATIC := libnearmem.a
LIB_SHARED := libnearmem.so
EXAMPLES := log_analyzer kv_cache_tier tiled_convolution tiled_matmul tiletrace \
            gpucpu_demo gpufpga_demo abyssal_demo

#==============================================================================
# TARGETS
#==============================================================================
.PHONY: all lib examples clean install cuda-info

all: lib examples

lib: $(LIB_STATIC) $(LIB_SHARED)

examples: $(EXAMPLES)

# Show CUDA configuration (for debugging)
cuda-info:
	@echo "=== CUDA Configuration ==="
	@echo "CUDA_PATH:      $(CUDA_PATH)"
	@echo "NVCC:           $(NVCC)"
	@echo "CUDA_AVAILABLE: $(CUDA_AVAILABLE)"
	@echo "CUDA_LIB_DIR:   $(CUDA_LIB_DIR)"
	@echo "CUDA_ENABLED:   $(CUDA_ENABLED)"
	@echo "LDFLAGS:        $(LDFLAGS)"
	@echo "CFLAGS:         $(CFLAGS)"
	@echo ""
	@echo "=== Checking files ==="
	@ls -la $(NVCC) 2>/dev/null || echo "nvcc not found at $(NVCC)"
	@ls -la $(CUDA_LIB_DIR)/libcuda.so* 2>/dev/null || echo "libcuda.so not found in $(CUDA_LIB_DIR)"

#==============================================================================
# LIBRARY BUILD
#==============================================================================

# Static library
$(LIB_STATIC): $(LIB_OBJS) $(CUDA_OBJS)
	ar rcs $@ $^

# Shared library
$(LIB_SHARED): $(LIB_OBJS) $(CUDA_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# C objects
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA objects
ifdef CUDA_ENABLED
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
# Stub for when CUDA isn't available
src/nearmem_kernels.o src/gpucpu_kernels.o:
	@echo "WARNING: CUDA not available - creating stub object for $@"
	@echo "  - NVCC path checked: $(NVCC)"
	@echo "  - libcuda.so checked in: $(CUDA_LIB_PATHS)"
	@echo "  - Run 'make cuda-info' for details"
	@echo "void $$(basename $@ .o)_stub(void) {}" | $(CC) -x c -c - -o $@
endif

#==============================================================================
# EXAMPLES
#==============================================================================

log_analyzer: examples/log_analyzer.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS)

kv_cache_tier: examples/kv_cache_tier.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS) -lpthread

tiled_convolution: examples/tiled_convolution.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS) -lm

tiled_matmul: examples/tiled_matmul.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS) -lm

tiletrace: examples/tiletrace.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS) -lm

gpucpu_demo: examples/gpucpu_demo.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS)

gpufpga_demo: examples/gpufpga_demo.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS)

abyssal_demo: examples/abyssal_demo.o $(LIB_STATIC)
	$(CC) -o $@ $< -L. -lnearmem $(LDFLAGS)

#==============================================================================
# INSTALL / CLEAN
#==============================================================================

clean:
	rm -f src/*.o examples/*.o
	rm -f $(LIB_STATIC) $(LIB_SHARED)
	rm -f $(EXAMPLES)

PREFIX ?= /usr/local

install: lib
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIB_STATIC) $(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(PREFIX)/lib/
	install -m 644 include/nearmem.h $(PREFIX)/include/
	ldconfig || true

uninstall:
	rm -f $(PREFIX)/lib/$(LIB_STATIC)
	rm -f $(PREFIX)/lib/$(LIB_SHARED)
	rm -f $(PREFIX)/include/nearmem.h

#==============================================================================
# HELP
#==============================================================================

help:
	@echo "Near-Memory Computing Library"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build library and examples"
	@echo "  lib       - Build static and shared libraries"
	@echo "  examples  - Build example programs"
	@echo "  cuda-info - Show CUDA detection results (debugging)"
	@echo "  clean     - Remove build artifacts"
	@echo "  install   - Install to PREFIX (default: /usr/local)"
	@echo ""
	@echo "Environment:"
	@echo "  CUDA_PATH - CUDA toolkit path (auto-detected if not set)"
	@echo "  PREFIX    - Installation prefix"
ifdef CUDA_ENABLED
	@echo ""
	@echo "CUDA: ENABLED"
	@echo "  Toolkit: $(CUDA_PATH)"
	@echo "  NVCC:    $(NVCC)"
	@echo "  Libs:    $(CUDA_LIB_DIR)"
else
	@echo ""
	@echo "CUDA: DISABLED (GPU acceleration not available)"
	@echo "  Run 'make cuda-info' for diagnostics"
endif
