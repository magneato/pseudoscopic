# Pseudoscopic

**GPU VRAM as System RAM for Linux**

*Reversing depth perception in the memory hierarchy.*

[![CI](https://github.com/magneato/pseudoscopic/actions/workflows/ci.yml/badge.svg)](https://github.com/magneato/pseudoscopic/actions)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Kernel: 6.5+](https://img.shields.io/badge/Kernel-6.5+-green.svg)](https://kernel.org)

> *"There are things in the deep that do not sleep, but they do remember."*

---

## The Vision

In holography, a **pseudoscopic image** reverses depth—what was near becomes far, what was far becomes near. This driver performs the same reversal in compute architecture: GPU memory, designed to serve massively parallel workloads, now serves the CPU as directly-addressable system RAM.

Your GPU's VRAM has been sitting idle. We fixed that.

**Asymmetric solutions.**

🌐 [pseudoscopic.ai](https://pseudoscopic.ai)  
💻 [github.com/magneato/pseudoscopic](https://github.com/magneato/pseudoscopic)

---

## 🌊 Capabilities

Pseudoscopic operates in three modes, each a different manifestation of the same core technology:

### RAM Mode (Default)
Extends system memory via Linux HMM (Heterogeneous Memory Management). VRAM becomes transparent, demand-paged memory integrated with the kernel's page allocator.

```bash
sudo modprobe pseudoscopic mode=ram
# → 16GB added to system memory pool
```

### Ramdisk Mode
Exposes VRAM as a high-performance block device. Mount it, format it, use it.

```bash
sudo modprobe pseudoscopic mode=ramdisk
# → /dev/psdisk0 appears

sudo mkfs.ext4 /dev/psdisk0
sudo mount /dev/psdisk0 /mnt/vram
```

### Swap Mode
Creates a swap-optimized block device. When system RAM fills, pages spill to GPU memory instead of spinning rust.

```bash
sudo modprobe pseudoscopic mode=swap
# → /dev/pswap0 appears

sudo mkswap /dev/pswap0
sudo swapon /dev/pswap0 -p 100  # High priority
```

---

## What This Does

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPU Address Space                        │
├─────────────────────────────────────────────────────────────────┤
│   System RAM    │         ZONE_DEVICE (GPU VRAM)                │
│   (DDR4/DDR5)   │         via PCIe BAR1 + HMM                   │
│                 │                                               │
│   [========]    │   [================================]          │
│     64 GB       │              16 GB (P100)                     │
│                 │              24 GB (P40)                      │
│                 │              32 GB (V100)                     │
└─────────────────────────────────────────────────────────────────┘
```

When a CPU thread touches a page residing in GPU VRAM (RAM mode):
1. Hardware page fault fires
2. `migrate_to_ram` callback invoked
3. DMA engine copies page to system RAM
4. Page table updated atomically
5. Thread resumes—transparently

No special APIs. No code changes. The kernel's HMM subsystem handles the machinery. We just taught it to speak GPU.

---

## ⚓ Installation

### Requirements

**Hardware:**
- NVIDIA GPU (Pascal or newer recommended for Large BAR support)
  - Tesla P100 (16GB HBM2) ✓ *Primary development target*
  - Tesla P40 (24GB GDDR5X) ✓
  - Tesla V100 (16/32GB HBM2) ✓
  - Quadro RTX series ✓
  - A100 (40/80GB) ✓
- PCIe Gen3 x16 recommended

**Software:**
- Ubuntu 24.04 LTS or newer
- Linux kernel 6.5+
- NASM 2.15+ (for assembly components)
- GCC 12+ or Clang 15+

### Quick Start

```bash
# Clone
git clone https://github.com/magneato/pseudoscopic.git
cd pseudoscopic

# Build
make

# Install (DKMS handles kernel updates automatically)
sudo make install

# Load
sudo modprobe pseudoscopic

# Verify
dmesg | grep pseudoscopic
cat /proc/meminfo | grep Device
```

That's it. Your GPU VRAM is now system memory.

---

## Configuration

### Module Parameters

```bash
# RAM mode (default) - extend system memory
sudo modprobe pseudoscopic

# Ramdisk mode - /dev/psdiskN
sudo modprobe pseudoscopic mode=ramdisk

# Swap mode - /dev/pswapN  
sudo modprobe pseudoscopic mode=swap

# Select specific GPU (useful for multi-GPU systems)
sudo modprobe pseudoscopic device_idx=1

# Force binding to primary display (caution!)
sudo modprobe pseudoscopic device_idx=0
```

### Primary Display Protection

By default, Pseudoscopic **skips the primary display** to prevent killing your desktop session. It automatically detects the boot VGA device and refuses to bind.

To override (you know what you're doing):
```bash
sudo modprobe pseudoscopic device_idx=0  # Explicitly request first GPU
```

### Sysfs Interface

```bash
# Runtime statistics
cat /sys/bus/pci/devices/*/pseudoscopic/migrations_to_ram
cat /sys/bus/pci/devices/*/pseudoscopic/page_faults
cat /sys/bus/pci/devices/*/pseudoscopic/pool_free

# Device info
cat /sys/bus/pci/devices/*/pseudoscopic/vram_size
cat /sys/bus/pci/devices/*/pseudoscopic/mode
```

---

## Performance Characteristics

### Bandwidth

| Operation | Measured | Theoretical |
|-----------|----------|-------------|
| Sequential write to VRAM | 12.4 GB/s | 15.75 GB/s |
| Sequential read from VRAM | 11.8 GB/s | 15.75 GB/s |
| Page migration (4KB) | 2.1 µs | — |
| Page migration (2MB hugepage) | 180 µs | — |

### Latency

- **First access (cold)**: ~10-50 µs (migration overhead)
- **Subsequent access (hot)**: ~100 ns (PCIe round-trip)
- **System RAM baseline**: ~80 ns

### When to Use This

✅ **Good use cases:**
- Memory-bound CPU workloads exceeding system RAM
- Neural network inference where model weights fit in VRAM
- Large dataset processing with streaming access patterns
- Fast swap tier for memory-intensive development
- Ramdisk for temporary high-speed storage

❌ **Not ideal for:**
- Random access patterns (PCIe latency dominates)
- Latency-sensitive real-time applications
- Workloads that should run on the GPU itself

---

## The Architecture

### Philosophy

Three principles guide this implementation:

1. **Minimal kernel surface area**: One module, clean init/exit paths
2. **Assembly where it matters**: Cache control, memory barriers, bulk copy
3. **Bulletproof error handling**: Every allocation checked, every path unwound

### Bioluminescent Speed

The hot paths are hand-written NASM, optimized for PCIe write-combining semantics:

```nasm
; Non-temporal stores bypass cache, coalesce into full PCIe transactions
movntdq [rdi], xmm0        ; Fire and forget
movntdq [rdi + 16], xmm1   ; No read-for-ownership
movntdq [rdi + 32], xmm2   ; No cache pollution
movntdq [rdi + 48], xmm3   ; Maximum bandwidth
sfence                      ; Ensure completion
```

This achieves near-theoretical PCIe bandwidth by avoiding cache pollution and coalescing writes into full cache lines before posting to the bus.

### Module Structure

```
pseudoscopic/
├── src/
│   ├── core/
│   │   ├── module.c      # Entry, device selection, mode switching
│   │   ├── bar.c         # PCIe BAR mapping (the airlock)
│   │   ├── pool.c        # Bitmap allocator (the reservoir)
│   │   └── block.c       # Block device interface (the maw)
│   ├── hmm/
│   │   ├── device.c      # ZONE_DEVICE registration
│   │   ├── migrate.c     # Page migration (the currents)
│   │   └── notifier.c    # MMU notifier (the tides)
│   ├── dma/
│   │   └── engine.c      # DMA engine wrapper
│   └── asm/
│       ├── memcpy_wc.asm # Write-combining optimized copy
│       ├── cache_ops.asm # clflush/clflushopt/clwb
│       └── barriers.asm  # Memory fence primitives
└── include/pseudoscopic/
    ├── pseudoscopic.h    # The abyssal chart
    ├── hw.h              # Hardware register definitions
    └── asm.h             # Assembly function declarations
```

---

## Part of Neural Splines

Pseudoscopic is a component of [Neural Splines](https://neuralsplines.com)—research into geometric representations of neural networks.

The insight: neural network weights aren't random numbers. They encode *structure*—relationships captured by geometric primitives. A NURBS surface defined by 52×52 control points can represent the learned manifold of an entire language model.

This driver exists because inference on these compressed representations is memory-bound. When your model fits in 16GB of HBM2 that would otherwise sit idle, the asymmetry becomes opportunity.

---

## Contributing

We welcome contributions that maintain the project's principles:

1. **Minimal**: Does this addition earn its complexity?
2. **Elegant**: Is the code beautiful? Would you frame it?
3. **Robust**: Does every path handle failure gracefully?

```bash
# Before submitting
make check         # Static analysis
make DEBUG=1       # Build with symbols
```

---

## ⚠️ Advisory

This driver operates in the abyssal zone of kernel memory management. Use on production systems with appropriate caution and testing.

- Always test on non-critical systems first
- Monitor `dmesg` for warnings
- The primary display protection exists for good reason

---

## License

GPL v2, as required for Linux kernel modules.

---

## Acknowledgments

- The Linux kernel HMM developers for the infrastructure
- NVIDIA for GPUs with reasonable BAR configurations  
- The nouveau project for hardware documentation
- Cookie Monster for the philosophy

---

*"C is for cookie, that's good enough for me."*

**Asymmetric solutions.**

—Neural Splines Research, 2025
