```
    ██████╗ ███████╗███████╗██╗   ██╗██████╗  ██████╗ ███████╗ ██████╗ ██████╗ ██████╗ ██╗ ██████╗
    ██╔══██╗██╔════╝██╔════╝██║   ██║██╔══██╗██╔═══██╗██╔════╝██╔════╝██╔═══██╗██╔══██╗██║██╔════╝
    ██████╔╝███████╗█████╗  ██║   ██║██║  ██║██║   ██║███████╗██║     ██║   ██║██████╔╝██║██║     
    ██╔═══╝ ╚════██║██╔══╝  ██║   ██║██║  ██║██║   ██║╚════██║██║     ██║   ██║██╔═══╝ ██║██║     
    ██║     ███████║███████╗╚██████╔╝██████╔╝╚██████╔╝███████║╚██████╗╚██████╔╝██║     ██║╚██████╗
    ╚═╝     ╚══════╝╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚══════╝ ╚═════╝ ╚═════╝ ╚═╝     ╚═╝ ╚═════╝
                                                                                                 
                        ╔═══════════════════════════════════════════════════╗
                        ║   GPU VRAM as Near-Memory Computing Platform      ║
                        ║   ─────────────────────────────────────────────   ║
                        ║   Where data doesn't move. Computation arrives.   ║
                        ╚═══════════════════════════════════════════════════╝
```

[![CI](https://github.com/magneato/pseudoscopic/actions/workflows/ci.yml/badge.svg)](https://github.com/magneato/pseudoscopic/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

## ◈ The Philosophical Foundation

*There is a peculiar irony in modern computing.* We build elaborate highways to shuttle data between memory kingdoms—CPU RAM to GPU VRAM and back—spending more time in transit than transformation. The PCIe bus becomes a bottleneck not because it's slow (12-32 GB/s is respectable), but because we insist on using it when we needn't.

Pseudoscopic challenges this orthodoxy. It asks: *what if the data simply stayed where it was, and the computation came to it?*

The name itself—*pseudoscopic*—refers to an optical phenomenon where depth perception inverts. Near becomes far; far becomes near. In our case, GPU memory that seemed distant and inaccessible becomes *close*, *addressable*, *intimate*. The CPU can reach out and touch VRAM as easily as it touches its own heap.

> **Fun fact**: The term "pseudoscopic" was coined by Charles Wheatstone in 1852 while studying stereoscopic vision. He noticed that swapping left-right images created an inverted depth perception—a metaphor for what we do with memory hierarchies.

---

## ◈ What This Actually Does

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                                   User Space                                        │
│   ┌───────────────┐    ┌───────────────┐    ┌───────────────┐    ┌───────────────┐  │
│   │   gpuCPU      │    │   gpuFPGA     │    │   Abyssal     │    │  Your App     │  │
│   │ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │    │ ┌───────────┐ │  │
│   │ │ x86 emu   │ │    │ │ LUT/FF    │ │    │ │ Trace     │ │    │ │  Anything │ │  │
│   │ │ on GPU    │ │    │ │ Simulator │ │    │ │ Debugger  │ │    │ │  you want │ │  │
│   │ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ │    │ └───────────┘ │  │
│   └───────┬───────┘    └───────┬───────┘    └───────┬───────┘    └───────┬───────┘  │
├───────────┴────────────────────┴────────────────────┴────────────────────┴──────────┤
│                           libnearmem (C/C++ API)                                    │
│  ┌─────────────────────────────────────────────────────────────────────────────┐    │
│  │  nearmem_alloc()    nearmem_sync()    nearmem_histogram()    nearmem_find() │    │
│  │  nearmem_tile()     nearmem_gpu_*()   nearmem_transform()    nearmem_sort() │    │
│  └─────────────────────────────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                           Pseudoscopic Kernel Driver                                │
│                                                                                     │
│      ┌─────────────┐      ┌─────────────┐      ┌─────────────┐                      │
│      │ /dev/psdisk0│      │ /dev/psdisk1│      │ /dev/psdisk2│  ← Block devices     │
│      └──────┬──────┘      └──────┬──────┘      └──────┬──────┘    exposing VRAM     │
│             │                    │                    │                             │
├─────────────┴────────────────────┴────────────────────┴─────────────────────────────┤
│                           Hardware Layer                                            │
│                                                                                     │
│      ╔═══════════════╗      ╔═══════════════╗      ╔═══════════════╗                │
│      ║    GPU #0     ║      ║    GPU #1     ║      ║    GPU #2     ║                │
│      ║ ┌───────────┐ ║      ║ ┌───────────┐ ║      ║ ┌───────────┐ ║                │
│      ║ │   VRAM    │ ║      ║ │   VRAM    │ ║      ║ │   VRAM    │ ║                │
│      ║ │  via BAR1 │ ║      ║ │  via BAR1 │ ║      ║ │  via BAR1 │ ║                │
│      ║ └───────────┘ ║      ║ └───────────┘ ║      ║ └───────────┘ ║                │
│      ╚═══════════════╝      ╚═══════════════╝      ╚═══════════════╝                │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

In more concrete terms:
- **Zero-Copy Access**: CPU directly reads/writes GPU VRAM through the PCIe BAR1 aperture
- **Near-Memory Computing**: Process terabytes where they live, not where you happen to compute
- **Multi-GPU Orchestration**: Enumerate, select, and manage multiple GPUs seamlessly
- **Memory Location Awareness**: Query any pointer to discover its true home—GPU or CPU

> **Historical aside**: The BAR1 (Base Address Register 1) aperture has existed since the earliest PCI graphics cards. What's changed is its *size*—modern GPUs with Large BAR support can expose their entire VRAM (32GB, 48GB, 80GB) through this window. We're not inventing new hardware; we're finally using 25 years of capability.

---

## ◈ The Quick Path to Enlightenment

### Installation

```bash
# Clone the repository (the source of truth)
git clone https://github.com/magneato/pseudoscopic.git
cd pseudoscopic

# The setup script handles everything—dependencies, compilation, installation
# It's opinionated because good defaults matter
sudo ./setup.sh

# Verify the universe is correctly configured
pseudoscopic-info
```

### What You Should See

```
    ╔════════════════════════════════════════════════════════════════════╗
    ║              Pseudoscopic System Information                       ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║  Driver Status                                                     ║
    ║    Module:  LOADED                                                 ║
    ║    Version: 0.2.0                                                  ║
    ║                                                                    ║
    ║  Block Devices                                                     ║
    ║    /dev/psdisk0: 16384 MB (GPU 0)                                  ║
    ║    /dev/psdisk1: 24576 MB (GPU 1)                                  ║
    ║                                                                    ║
    ║  NVIDIA GPUs Detected                                              ║
    ║    [0] GeForce RTX 4090      │ 24 GB VRAM │ BAR1: Full             ║
    ║    [1] Tesla V100-PCIE       │ 32 GB VRAM │ BAR1: Full             ║
    ╚════════════════════════════════════════════════════════════════════╝
```

---

## ◈ Code That Speaks for Itself

### The C Path (For Those Who Appreciate Simplicity)

```c
#include <nearmem/nearmem.h>
#include <nearmem/nearmem_gpu.h>

int main(void) {
    nearmem_ctx_t ctx;
    nearmem_region_t region;
    
    // Initialize—this opens /dev/psdisk0 and establishes the CUDA context
    // The "0" selects GPU index 0; adjust for multi-GPU systems
    nearmem_init(&ctx, "/dev/psdisk0", 0);
    
    // Allocate one billion bytes in VRAM
    // After this call, region.cpu_ptr is a valid pointer to GPU memory
    nearmem_alloc(&ctx, &region, 1ULL << 30);
    
    // The magic: CPU writes directly to GPU memory
    // No cudaMemcpy. No staging buffer. Just pointer arithmetic.
    float *data = (float*)region.cpu_ptr;
    for (size_t i = 0; i < 256 * 1024 * 1024; i++) {
        data[i] = (float)i * 3.14159f;  // Pi, approximately
    }
    
    // Tell the GPU our writes are complete
    // This flushes write-combine buffers and ensures coherency
    nearmem_sync(&ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    // The region now contains 1GB of floats, accessible by CUDA kernels
    // Your kernel sees region.gpu_ptr as a device pointer
    
    // Cleanup—because responsible programmers clean up
    nearmem_free(&ctx, &region);
    nearmem_shutdown(&ctx);
    return 0;
}
```

### The C++ Path (For Those Who Appreciate RAII)

```cpp
#include <nearmem/nearmem.hpp>
#include <iostream>
#include <numeric>

int main() {
    // Context manages device lifetime; destructor cleans up
    nearmem::Context ctx(0);  // GPU 0
    
    // Type-safe allocation with usage hints
    // The usage hint optimizes internal behavior (caching, prefetch, etc.)
    auto buffer = ctx.alloc<float>(
        256 * 1024 * 1024,  // 256 million floats = 1 GB
        nearmem::Usage::STREAMING | nearmem::Usage::WRITE_MOSTLY
    );
    
    // Container semantics—iterators, range-for, the works
    std::iota(buffer.begin(), buffer.end(), 0.0f);
    
    // Explicit sync—we don't hide the memory model
    buffer.sync_to_gpu();
    
    // Location awareness built-in
    std::cout << "Buffer location: " 
              << (buffer.is_gpu_memory() ? "VRAM" : "RAM")
              << '\n';
    
    return 0;  // buffer destructor frees VRAM
}
```

### Compiling Your Vision

```bash
# The pkg-config way (recommended)
gcc -o myprogram myprogram.c $(pkg-config --cflags --libs nearmem)
g++ -std=c++17 -o myprogram myprogram.cpp $(pkg-config --cflags --libs nearmem)

# The explicit way (for the curious)
gcc -o myprogram myprogram.c -I/usr/local/include -L/usr/local/lib -lnearmem -lcuda -ldl
```

---

## ◈ Capabilities Explored

### Multi-GPU Navigation

Modern workstations sport multiple GPUs—training rigs, rendering farms, research clusters. Pseudoscopic treats them as first-class citizens:

```c
// Discover the fleet
int count = nearmem_gpu_count();
printf("Found %d GPU(s)\n", count);

for (int i = 0; i < count; i++) {
    nearmem_gpu_info_t info;
    nearmem_gpu_get_info(i, &info);
    
    printf("GPU %d: %s\n", i, info.name);
    printf("  PCI Address: %s\n", info.pci_address);
    printf("  VRAM Total:  %lu MB\n", info.vram_size >> 20);
    printf("  VRAM Free:   %lu MB\n", info.vram_available >> 20);
    printf("  BAR1 Base:   0x%lx\n", info.bar1_base);
    printf("  Block Dev:   %s\n", info.block_device);
    printf("  HMM Support: %s\n", info.supports_hmm ? "yes" : "no");
    printf("\n");
}
```

### Memory Location Oracle

Where does a pointer live? This question seems trivial until you're debugging a segfault at 2 AM and nothing makes sense:

```c
void investigate(void *ptr) {
    nearmem_memloc_info_t info;
    nearmem_memloc_t loc = nearmem_get_memloc(ptr, &info);
    
    switch (loc) {
        case MEMLOC_GPU_VRAM:
            printf("Pointer %p lives in GPU %d VRAM at offset 0x%lx\n",
                   ptr, info.gpu_index, info.gpu_offset);
            break;
        case MEMLOC_CPU:
            printf("Pointer %p is in system RAM\n", ptr);
            break;
        case MEMLOC_PINNED:
            printf("Pointer %p is pinned CPU memory (fast GPU access)\n", ptr);
            break;
        case MEMLOC_MAPPED:
            printf("Pointer %p is memory-mapped I/O\n", ptr);
            break;
        default:
            printf("Pointer %p is of unknown origin\n", ptr);
    }
}
```

> **Why this matters**: Traditional debuggers can't distinguish GPU pointers from garbage. When a `cudaMemcpy` fails silently, knowing the *location* of your data is half the battle.

---

## ◈ The Example Menagerie

| Program | What It Demonstrates |
|---------|----------------------|
| `log_analyzer` | Zero-copy grep on VRAM-resident logs—search 50 GB in 200ms |
| `kv_cache_tier` | LLM key-value cache spillover to VRAM when CPU RAM fills |
| `tiled_convolution` | Image convolution with stencil halos, tiled for cache efficiency |
| `tiled_matmul` | Cache-blocked SGEMM, demonstrating tiled memory patterns |
| `tiletrace` | Procedural ray-traced flight simulator with GPU compute |
| `gpucpu_demo` | x86 CPU emulation running on GPU—because why not |
| `gpufpga_demo` | FPGA gate-level simulation with branchless LUT evaluation |
| `abyssal_demo` | Interactive circuit debugger with waveform capture |

```bash
# Run from the examples directory
cd /usr/local/share/pseudoscopic/examples
./gpufpga_demo      # Watch an FPGA simulation unfold
./abyssal_demo      # Explore circuits interactively
```

---

## ◈ System Requirements

### Hardware Prerequisites

- **NVIDIA GPU** with BAR1 aperture (most consumer and professional GPUs since 2016)
- **Large BAR enabled** in BIOS for maximum VRAM exposure
- Tested configurations:
  - GeForce RTX 2080/3080/4090
  - Tesla P100, V100, A100
  - Quadro RTX 4000/6000/8000
  - RTX A4000/A5000/A6000

> **The Large BAR story**: Originally, PCIe BARs were limited to 256 MB due to 32-bit address space constraints. "Above 4G Decoding" and "Resizable BAR" BIOS settings unlock the full aperture. If you see only 256 MB, check your BIOS.

### Software Prerequisites

- Linux kernel 5.4 or later (HMM support matured here)
- GCC version matching your kernel's build compiler
- CUDA Toolkit 11.0+ (optional, but enables GPU-side operations)
- NASM 2.14+ (for assembly optimizations)

### Checking Compiler Compatibility

The kernel and module *must* be compiled with the same GCC major version:

```bash
# What did the kernel expect?
cat /proc/version | grep -oP 'gcc[- ]version \K[0-9]+\.[0-9]+'

# What do you have?
gcc --version | head -1

# Install matching version if needed
sudo apt install gcc-13 g++-13
```

---

## ◈ Building from Source

### The Comprehensive Path

```bash
# Full installation: kernel module + library + examples + tools
sudo ./setup.sh

# The setup script will:
# 1. Detect your kernel's compiler
# 2. Install the matching GCC if needed
# 3. Build the kernel module
# 4. Register with DKMS for automatic rebuilds
# 5. Build libnearmem with CUDA support
# 6. Install examples and documentation
```

### The Surgical Path

```bash
# Just the kernel module
make modules
sudo make install

# Just the library
cd contrib/nearmem
make lib

# Just the examples
make examples

# With specific CUDA path
make CUDA_PATH=/usr/local/cuda-12.3 lib
```

### Build Customization

```bash
# Debug build with symbols
make DEBUG=1 lib

# Specific compiler
make CC=gcc-13 CXX=g++-13 lib

# Verbose make output
make V=1 modules
```

---

## ◈ API Reference (Abbreviated)

### Core Lifecycle

```c
nearmem_error_t nearmem_init(nearmem_ctx_t *ctx, const char *device, int cuda_dev);
nearmem_error_t nearmem_init_auto(nearmem_ctx_t *ctx);  // Auto-detect first device
void nearmem_shutdown(nearmem_ctx_t *ctx);
size_t nearmem_get_capacity(nearmem_ctx_t *ctx);
```

### Memory Management

```c
nearmem_error_t nearmem_alloc(nearmem_ctx_t *ctx, nearmem_region_t *region, size_t size);
void nearmem_free(nearmem_ctx_t *ctx, nearmem_region_t *region);
nearmem_error_t nearmem_map_offset(nearmem_ctx_t *ctx, nearmem_region_t *region,
                                    uint64_t offset, size_t size);
```

### Coherency Control

```c
nearmem_error_t nearmem_sync(nearmem_ctx_t *ctx, nearmem_sync_t direction);
nearmem_error_t nearmem_sync_region(nearmem_ctx_t *ctx, nearmem_region_t *region,
                                     nearmem_sync_t direction);
// Where direction is: NEARMEM_SYNC_CPU_TO_GPU | NEARMEM_SYNC_GPU_TO_CPU | NEARMEM_SYNC_FULL
```

### GPU Fleet Management

```c
int nearmem_gpu_count(void);
int nearmem_gpu_enumerate(nearmem_gpu_info_t *infos, int max);
int nearmem_gpu_get_info(int index, nearmem_gpu_info_t *info);
int nearmem_gpu_select(int index);
bool nearmem_is_gpu_memory(const void *ptr);
int nearmem_get_gpu_for_ptr(const void *ptr);
```

---

## ◈ Contributing

We welcome contributions that align with our philosophy: **minimal surface area, bulletproof reliability, beauty in code**. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

Areas of interest:
- Testing on exotic hardware configurations
- Performance optimization (especially assembly hot paths)
- Documentation improvements
- Consumer GPU resizable BAR support

---

## ◈ Citation

If Pseudoscopic contributes to your research, we'd appreciate a citation:

```bibtex
@software{pseudoscopic2026,
  title     = {Pseudoscopic: Near-Memory Computing via GPU VRAM},
  author    = {{Neural Splines LLC}},
  year      = {2026},
  url       = {https://github.com/magneato/pseudoscopic},
  note      = {Accessed: 2026}
}
```

---

## ◈ License

MIT License—use freely, modify boldly, attribute kindly.

---

```
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║      "The GPU is not an accelerator attached to your computer.                ║
    ║       The GPU IS the computer. The CPU merely orchestrates."                  ║
    ║                                                                               ║
    ║                                        — Neural Splines Research, 2026        ║
    ║                                           Asymmetric Solutions                ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
```

*Where data rests, computation arrives.* 🍪
