# Pseudoscopic

**GPU VRAM as Near-Memory Computing Platform**

[![CI](https://github.com/neuralsplines/pseudoscopic/actions/workflows/ci.yml/badge.svg)](https://github.com/neuralsplines/pseudoscopic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## Overview

Pseudoscopic transforms NVIDIA GPU VRAM into a unified memory platform accessible by both CPU and GPU without expensive data copies. This enables:

- **Zero-Copy Access**: CPU directly reads/writes GPU VRAM via BAR1 aperture
- **Near-Memory Computing**: Process data where it lives, not where you compute
- **Multi-GPU Support**: Enumerate and manage multiple GPUs
- **Memory Location API**: Detect if any pointer is in GPU or CPU memory

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Applications                                     │
│   ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                │
│   │  gpuCPU      │  │  gpuFPGA     │  │  Abyssal     │                │
│   │ (x86 emu)    │  │ (FPGA sim)   │  │ (debugger)   │                │
│   └──────────────┘  └──────────────┘  └──────────────┘                │
├────────────────────────────────────────────────────────────────────────┤
│                      C++ Wrapper (nearmem.hpp)                         │
│              RAII • Usage Hints • Type Safety • C++17                  │
├────────────────────────────────────────────────────────────────────────┤
│                      Near-Memory Library (libnearmem)                  │
│      ┌────────────────────────────────────────────────────────────────┐│
│      │  Core API    │  Tiled API   │  GPU API    │  Sync API         ││
│      └────────────────────────────────────────────────────────────────┘│
├────────────────────────────────────────────────────────────────────────┤
│                      Pseudoscopic Driver (pseudoscopic.ko)             │
│                   /dev/psdisk0  /dev/psdisk1  ...                      │
├────────────────────────────────────────────────────────────────────────┤
│                      GPU Hardware (NVIDIA)                             │
│                      BAR1 Aperture → VRAM                              │
└────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/neuralsplines/pseudoscopic.git
cd pseudoscopic

# Run installer (requires root)
sudo ./setup.sh

# Verify installation
pseudoscopic-info
```

### Check System Status

```bash
$ pseudoscopic-info

Pseudoscopic System Information
================================

Driver Status:
  Module:  LOADED
  version: 0.0.1

Block Devices:
  /dev/psdisk0: 16384 MB
  /dev/psdisk1: 16384 MB

NVIDIA GPUs:
  01:00.0 VGA compatible controller: NVIDIA Corporation Device [10de:2204]
```

### C Example

```c
#include <nearmem/nearmem.h>
#include <nearmem/nearmem_gpu.h>

int main() {
    // Initialize for first GPU
    nearmem_ctx_t ctx;
    nearmem_init(&ctx, "/dev/psdisk0", 0);
    
    // Allocate 1GB in VRAM
    nearmem_region_t region;
    nearmem_alloc(&ctx, &region, 1ULL << 30);
    
    // CPU can directly access GPU memory!
    float *data = (float*)region.cpu_ptr;
    for (int i = 0; i < 1000; i++) {
        data[i] = i * 3.14f;
    }
    
    // Sync to GPU
    nearmem_sync(&ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    // Check memory location
    if (nearmem_is_gpu_memory(data)) {
        printf("Data is in GPU VRAM\n");
    }
    
    // Cleanup
    nearmem_free(&ctx, &region);
    nearmem_shutdown(&ctx);
    return 0;
}
```

### C++ Example

```cpp
#include <nearmem/nearmem.hpp>

int main() {
    // RAII context - automatically cleans up
    nearmem::Context ctx(0);  // GPU index 0
    
    // Typed allocation with usage hints
    auto buffer = ctx.alloc<float>(1'000'000, 
        nearmem::Usage::STREAMING | nearmem::Usage::GPU_ONLY);
    
    // Standard container-like access
    buffer[0] = 3.14f;
    buffer.fill(0.0f);
    
    // Range-based iteration
    for (auto& val : buffer) {
        val = 42.0f;
    }
    
    // Explicit sync
    buffer.sync_to_gpu();
    
    // Check location
    if (buffer.is_gpu_memory()) {
        std::cout << "Buffer is in VRAM\n";
    }
    
    return 0;  // Automatic cleanup
}
```

### Compile with pkg-config

```bash
# C program
gcc myprogram.c $(pkg-config --cflags --libs nearmem) -o myprogram

# C++ program  
g++ -std=c++17 myprogram.cpp $(pkg-config --cflags --libs nearmem) -o myprogram
```

## Features

### Multi-GPU Support

```c
// Enumerate all GPUs
int count = nearmem_gpu_count();

for (int i = 0; i < count; i++) {
    nearmem_gpu_info_t info;
    nearmem_gpu_get_info(i, &info);
    
    printf("GPU %d: %s\n", i, info.name);
    printf("  VRAM: %lu MB\n", info.vram_size >> 20);
    printf("  BAR1: 0x%lx\n", info.bar1_base);
    printf("  Device: %s\n", info.block_device);
}
```

### Memory Location Detection

```c
void *ptr = some_allocation();

switch (nearmem_get_memloc(ptr, NULL)) {
    case MEMLOC_GPU_VRAM:
        printf("Pointer is in GPU VRAM\n");
        break;
    case MEMLOC_CPU:
        printf("Pointer is in system RAM\n");
        break;
    case MEMLOC_MAPPED:
        printf("Pointer is memory-mapped\n");
        break;
}
```

### Usage-Based Allocation (C++)

```cpp
// Streaming workload (large sequential access)
auto stream_buf = ctx.alloc<char>(1_GB, Usage::STREAMING);

// Random access pattern  
auto lookup_buf = ctx.alloc<Entry>(1_M, Usage::RANDOM);

// Tiled computation
auto tile_buf = ctx.alloc<float>(64*64, Usage::TILED);

// Double-buffered pipeline
auto ping = ctx.alloc<Data>(size, Usage::DOUBLE_BUF);
auto pong = ctx.alloc<Data>(size, Usage::DOUBLE_BUF);
```

## Example Programs

| Program | Description |
|---------|-------------|
| `log_analyzer` | Zero-copy grep on VRAM-resident logs |
| `kv_cache_tier` | LLM KV-cache tiering to VRAM |
| `tiled_convolution` | Image convolution with stencil halos |
| `tiled_matmul` | Cache-blocked matrix multiplication |
| `tiletrace` | Procedural ray-traced flight simulator |
| `gpucpu_demo` | x86 emulation running on GPU |
| `gpufpga_demo` | FPGA simulation with branchless LUTs |
| `abyssal_demo` | Interactive circuit debugger |

Run examples:
```bash
cd /usr/local/share/pseudoscopic/examples
./abyssal_demo
```

## Requirements

### Hardware
- NVIDIA GPU with BAR1 aperture (most desktop/server GPUs)
- Tested: Tesla P100, V100, A100, RTX 20xx/30xx/40xx

### Software
- Linux kernel 5.4+
- GCC 7+ (matching kernel compiler for driver)
- CUDA toolkit (optional, for GPU-side operations)

### Compiler Compatibility

The kernel module **must** be compiled with the same GCC major version as your kernel:

```bash
# Check kernel compiler
cat /proc/version | grep -oP 'gcc[- ]version \K[0-9]+\.[0-9]+'

# Install matching GCC if needed
sudo apt install gcc-11
```

## Building from Source

```bash
# Full build (driver + library + examples)
sudo ./setup.sh

# Library only (no root required)
./setup.sh --lib-only

# Just the library with make
cd contrib/nearmem
make lib
make examples
```

### Build Options

```bash
# Debug build
make DEBUG=1 lib

# With CUDA support (if toolkit available)
make CUDA_PATH=/usr/local/cuda lib

# Specific GCC version (for kernel module)
make CC=gcc-11 lib
```

## API Reference

### Core Functions

```c
// Initialization
int nearmem_init(nearmem_ctx_t *ctx, const char *device, int flags);
void nearmem_shutdown(nearmem_ctx_t *ctx);

// Allocation
int nearmem_alloc(nearmem_ctx_t *ctx, nearmem_region_t *region, size_t size);
void nearmem_free(nearmem_ctx_t *ctx, nearmem_region_t *region);

// Synchronization  
int nearmem_sync(nearmem_ctx_t *ctx, int direction);
```

### GPU Functions

```c
// Enumeration
int nearmem_gpu_count(void);
int nearmem_gpu_enumerate(nearmem_gpu_info_t *infos, int max);
int nearmem_gpu_get_info(int index, nearmem_gpu_info_t *info);

// Memory location
bool nearmem_is_gpu_memory(const void *ptr);
uint64_t nearmem_gpu_get_vram_base(int gpu_index);
uint64_t nearmem_gpu_get_vram_size(int gpu_index);
```

### Tiled Functions

```c
// Tile descriptor creation
nearmem_tile_desc_t nearmem_tile_desc_1d(size_t total, size_t tile);
nearmem_tile_desc_t nearmem_tile_desc_2d(size_t h, size_t w, size_t th, size_t tw);

// Tile operations  
int nearmem_tile_prefetch(nearmem_ctx_t *ctx, nearmem_region_t *r, int tile_idx);
void *nearmem_tile_ptr(nearmem_region_t *region, int tile_idx);
```

## License

MIT License - see [LICENSE](LICENSE)

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing`)
3. Commit changes (`git commit -am 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing`)
5. Open a Pull Request

## Citation

If you use Pseudoscopic in research, please cite:

```bibtex
@software{pseudoscopic,
  title = {Pseudoscopic: Near-Memory Computing via GPU VRAM},
  author = {Neural Splines LLC},
  year = {2025},
  url = {https://github.com/neuralsplines/pseudoscopic}
}
```

---

*"The GPU is not an accelerator. The GPU IS the computer."* 🍪
