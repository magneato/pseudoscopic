# Near-Memory Computing Library Makefile
#
# Builds the nearmem library and examples.
# Requires CUDA toolkit for GPU-accelerated operations.
#
# Copyright (C) 2025 Neural Splines LLC

# Detect CUDA installation
CUDA_PATH ?= /usr/local/cuda
NVCC := $(CUDA_PATH)/bin/nvcc

# Check if CUDA is available
CUDA_AVAILABLE := $(shell which $(NVCC) 2>/dev/null)

# Compiler settings
CC := gcc
CFLAGS := -Wall -Wextra -O2 -fPIC -I./include
LDFLAGS := -ldl -lpthread

# CUDA settings (if available)
ifdef CUDA_AVAILABLE
  NVCCFLAGS := -O2 -Xcompiler -fPIC -I./include
  LDFLAGS += -L$(CUDA_PATH)/lib64 -lcuda
  CFLAGS += -DNEARMEM_HAS_CUDA
endif

# Source files
LIB_SRCS := src/nearmem.c src/nearmem_tile.c src/gpucpu.c src/gpufpga.c src/abyssal.c src/nearmem_gpu.c
CUDA_SRCS := src/nearmem_kernels.cu
EXAMPLE_SRCS := examples/log_analyzer.c examples/kv_cache_tier.c examples/tiled_convolution.c examples/gpucpu_demo.c examples/gpufpga_demo.c examples/abyssal_demo.c

# Object files
LIB_OBJS := $(LIB_SRCS:.c=.o)
CUDA_OBJS := $(CUDA_SRCS:.cu=.o)
EXAMPLE_OBJS := $(EXAMPLE_SRCS:.c=.o)

# Output
LIB_STATIC := libnearmem.a
LIB_SHARED := libnearmem.so
EXAMPLES := log_analyzer

# Targets
.PHONY: all lib examples clean install

all: lib examples

lib: $(LIB_STATIC) $(LIB_SHARED)

examples: $(EXAMPLES)

# Static library
$(LIB_STATIC): $(LIB_OBJS) $(CUDA_OBJS)
	ar rcs $@ $^

# Shared library
$(LIB_SHARED): $(LIB_OBJS) $(CUDA_OBJS)
	$(CC) -shared -o $@ $^ $(LDFLAGS)

# C objects
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CUDA objects (if available)
ifdef CUDA_AVAILABLE
%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
else
# Stub for when CUDA isn't available
src/nearmem_kernels.o:
	@echo "CUDA not available, creating stub..."
	@echo "void nearmem_cuda_stub(void) {}" | $(CC) -x c -c - -o $@
endif

# Examples
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

EXAMPLES := log_analyzer kv_cache_tier tiled_convolution tiled_matmul tiletrace gpucpu_demo gpufpga_demo abyssal_demo

# Clean
clean:
	rm -f src/*.o examples/*.o
	rm -f $(LIB_STATIC) $(LIB_SHARED)
	rm -f $(EXAMPLES)

# Install
PREFIX ?= /usr/local

install: lib
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include
	install -m 644 $(LIB_STATIC) $(PREFIX)/lib/
	install -m 755 $(LIB_SHARED) $(PREFIX)/lib/
	install -m 644 include/nearmem.h $(PREFIX)/include/
	ldconfig || true

# Uninstall
uninstall:
	rm -f $(PREFIX)/lib/$(LIB_STATIC)
	rm -f $(PREFIX)/lib/$(LIB_SHARED)
	rm -f $(PREFIX)/include/nearmem.h

# Help
help:
	@echo "Near-Memory Computing Library"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build library and examples"
	@echo "  lib      - Build static and shared libraries"
	@echo "  examples - Build example programs"
	@echo "  clean    - Remove build artifacts"
	@echo "  install  - Install to PREFIX (default: /usr/local)"
	@echo ""
	@echo "Environment:"
	@echo "  CUDA_PATH - CUDA toolkit path (default: /usr/local/cuda)"
	@echo "  PREFIX    - Installation prefix"
ifdef CUDA_AVAILABLE
	@echo ""
	@echo "CUDA: $(NVCC)"
else
	@echo ""
	@echo "CUDA: Not found (GPU acceleration disabled)"
endif
