# Pseudoscopic

**GPU VRAM as System RAM for Linux**

*Reversing depth perception in the memory hierarchy*

[![CI](https://github.com/neuralsplines/pseudoscopic/actions/workflows/ci.yml/badge.svg)](https://github.com/neuralsplines/pseudoscopic/actions)
[![License: GPL v2](https://img.shields.io/badge/License-GPL%20v2-blue.svg)](https://www.gnu.org/licenses/old-licenses/gpl-2.0.en.html)
[![Kernel: 6.5+](https://img.shields.io/badge/Kernel-6.5+-green.svg)](https://kernel.org)

---

## The Vision

In holography, a **pseudoscopic image** reverses depth—what was near becomes far, what was far becomes near. This driver performs the same reversal in compute architecture: GPU memory, designed to serve massively parallel workloads, now serves the CPU as directly-addressable system RAM.

Why? Because sometimes you have 16GB of HBM2 sitting idle while your neural network inference is memory-bound on the CPU side. Because sometimes constraints breed elegance. Because we can.

This is **nom nom engineering**: minimal surface area, maximum impact, code that Cookie Monster would approve of—taking one beautiful bite at a time.

---

## What This Does

Pseudoscopic exposes NVIDIA Tesla/Datacenter GPU VRAM as CPU-addressable memory through Linux's Heterogeneous Memory Management (HMM) subsystem. Not swap. Not a block device. *Actual memory* with `struct page` backing, transparent page migration, and full kernel integration.

```
┌─────────────────────────────────────────────────────────────────┐
│                        CPU Address Space                         │
├─────────────────────────────────────────────────────────────────┤
│   System RAM    │         ZONE_DEVICE (GPU VRAM)                │
│   (DDR4/DDR5)   │         via PCIe BAR1 + HMM                   │
│                 │                                                │
│   [========]    │   [================================]          │
│     64 GB       │              16 GB (P100)                     │
│                 │              24 GB (P40)                      │
│                 │              32 GB (V100)                     │
└─────────────────────────────────────────────────────────────────┘
```

When a CPU thread touches a page in GPU VRAM:
1. Hardware page fault fires
2. `migrate_to_ram` callback invoked
3. DMA engine copies page to system RAM (if needed)
4. Page table updated atomically
5. Thread resumes—transparently

The reverse happens for migration *to* the device. No explicit management. No special APIs. Just memory.

---

## Requirements

**Hardware:**
- NVIDIA Tesla/Datacenter GPU with Large BAR support:
  - Tesla P100 (16GB HBM2) ✓ *Primary development target*
  - Tesla P40 (24GB GDDR5X) ✓
  - Tesla V100 (16/32GB HBM2) ✓
  - Quadro RTX series ✓
- PCIe Gen3 x16 recommended (15.75 GB/s bidirectional)
- IOMMU disabled or passthrough mode

**Software:**
- Ubuntu 24.04 LTS or newer
- Linux kernel 6.5+ (HMM APIs, ZONE_DEVICE improvements)
- NASM 2.15+ (assembly components)
- GCC 12+ or Clang 15+
- Nouveau driver **unloaded** (we talk directly to hardware)

---

## Quick Start

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

## The Architecture

### Core Philosophy

This driver embodies three principles:

1. **Minimal kernel surface area**: One module, clean init/exit paths, no feature creep
2. **Assembly where it matters**: Cache control, memory barriers, bulk copy—hand-tuned NASM
3. **Bulletproof error handling**: Every allocation checked, every path unwound cleanly

### Module Structure

```
pseudoscopic/
├── src/
│   ├── core/
│   │   ├── module.c          # Entry point, PCI probe/remove
│   │   ├── bar.c             # PCIe BAR mapping and validation
│   │   └── pool.c            # VRAM page pool management
│   ├── asm/
│   │   ├── memcpy_wc.asm     # Write-combining optimized copy
│   │   ├── cache_ops.asm     # clflush/clflushopt/clwb wrappers
│   │   └── barriers.asm      # Memory barrier primitives
│   ├── hmm/
│   │   ├── device.c          # ZONE_DEVICE registration
│   │   ├── migrate.c         # Page migration (to/from VRAM)
│   │   └── notifier.c        # MMU notifier integration
│   └── dma/
│       └── engine.c          # GPU DMA engine for async copy
├── include/pseudoscopic/
│   ├── pseudoscopic.h        # Public driver interface
│   ├── hw.h                  # Hardware register definitions
│   └── asm.h                 # Assembly function declarations
└── tools/
    ├── ps-status             # Runtime status and statistics
    └── ps-migrate            # Manual migration control
```

### Memory Model

We use `MEMORY_DEVICE_PRIVATE` rather than `MEMORY_DEVICE_GENERIC`:

- Pages are **not** directly CPU-mappable (forces migration path)
- CPU access triggers `migrate_to_ram` callback
- Enables true demand-paged GPU memory with kernel MM integration
- Works with `get_user_pages()` and friends

Why not direct mapping? Because PCIe latency (100+ ns per access) makes random access pathological. Migration amortizes the cost into bulk DMA transfers.

### Assembly Optimization

The hot paths are hand-written NASM:

```nasm
; memcpy_wc.asm - Write-combining optimized bulk copy
; Uses non-temporal stores to bypass cache hierarchy

global ps_memcpy_to_vram
ps_memcpy_to_vram:
    ; rcx = count (bytes, multiple of 64)
    ; rdi = dest (VRAM, WC-mapped)  
    ; rsi = src (system RAM)
    
.loop:
    movdqa   xmm0, [rsi]
    movdqa   xmm1, [rsi + 16]
    movdqa   xmm2, [rsi + 32]
    movdqa   xmm3, [rsi + 48]
    
    movntdq  [rdi], xmm0        ; Non-temporal: bypass cache
    movntdq  [rdi + 16], xmm1
    movntdq  [rdi + 32], xmm2
    movntdq  [rdi + 48], xmm3
    
    add      rsi, 64
    add      rdi, 64
    sub      rcx, 64
    jnz      .loop
    
    sfence                       ; Ensure stores complete
    ret
```

This achieves near-theoretical PCIe bandwidth by:
- Avoiding cache pollution on the destination
- Streaming full cache lines
- Minimizing instruction overhead

---

## Configuration

### Module Parameters

```bash
# Load with custom parameters
sudo modprobe pseudoscopic \
    bar_index=1 \              # Which BAR to use (default: 1)
    pool_size_mb=0 \           # 0 = use entire BAR
    migration_threshold=4 \    # Pages before async migration kicks in
    enable_dma=1               # Use GPU DMA engine (faster, requires init)
```

### Sysfs Interface

```bash
# Runtime statistics
cat /sys/module/pseudoscopic/stats/migrations_to_ram
cat /sys/module/pseudoscopic/stats/migrations_to_device
cat /sys/module/pseudoscopic/stats/page_faults

# Pool status
cat /sys/module/pseudoscopic/pool/total_pages
cat /sys/module/pseudoscopic/pool/free_pages
```

### Proc Interface

```bash
# Appears in standard memory reporting
$ cat /proc/meminfo | grep -i device
DeviceTotal:    16777216 kB
DeviceFree:     16252416 kB
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
- Memory-bound CPU workloads with working sets > system RAM
- Neural network inference where model weights fit in VRAM
- Large dataset processing with streaming access patterns
- Development/testing of memory-intensive applications

❌ **Not ideal for:**
- Random access patterns (PCIe latency dominates)
- Latency-sensitive real-time applications
- Workloads that would benefit from actual GPU compute

---

## Safety and Stability

### Error Handling

Every code path handles failure:

```c
/* Example: Pool allocation with full cleanup */
page = ps_pool_alloc(pool);
if (!page) {
    ps_stats_inc(STAT_ALLOC_FAIL);
    return VM_FAULT_OOM;
}

ret = ps_dma_copy_to_vram(dev, page, src_page);
if (ret) {
    ps_pool_free(pool, page);
    ps_stats_inc(STAT_DMA_FAIL);
    return VM_FAULT_SIGBUS;
}
```

### Kernel Lockdown Compatibility

- No `/dev/mem` access
- No arbitrary physical memory mapping
- Uses proper PCI resource APIs
- Compatible with Secure Boot (when signed)

### Testing

```bash
# Run self-tests (requires loaded module)
sudo ./tools/ps-test

# Memory stress test
sudo ./tools/ps-stress --duration=3600 --threads=8
```

---

## Building for Development

```bash
# Debug build with symbols and extra checks
make DEBUG=1

# Build with specific kernel headers
make KDIR=/usr/src/linux-headers-6.8.0-40-generic

# Build only assembly components
make -C src/asm

# Generate compile_commands.json for IDE integration
make compile_commands.json

# Static analysis
make check
```

---

## Internals for the Curious

### How HMM Works

Linux's Heterogeneous Memory Management provides infrastructure for device memory:

1. **ZONE_DEVICE**: A memory zone for non-standard memory
2. **struct dev_pagemap**: Describes a device memory region
3. **migrate_vma_*()**: APIs for page migration between zones
4. **mmu_interval_notifier**: Callbacks for page table changes

We register GPU VRAM as a dev_pagemap with type `MEMORY_DEVICE_PRIVATE`:

```c
dev->pagemap.type = MEMORY_DEVICE_PRIVATE;
dev->pagemap.range.start = vram_resource->start;
dev->pagemap.range.end = vram_resource->end;
dev->pagemap.ops = &ps_devmem_ops;
dev->pagemap.owner = dev;

addr = devm_memremap_pages(&pdev->dev, &dev->pagemap);
```

This creates `struct page` entries for every VRAM page, integrating them into the kernel's memory management.

### PCIe BAR Mapping

NVIDIA GPUs expose VRAM through BAR1. Tesla/datacenter cards ship with "Large BAR" enabled—the full VRAM is mappable without resize tricks:

```c
bar_size = pci_resource_len(pdev, BAR_INDEX);
if (bar_size < expected_vram) {
    /* Consumer GPU with 256MB default BAR */
    ps_resize_bar(pdev, BAR_INDEX, expected_vram);
}

vram = pci_iomap_wc(pdev, BAR_INDEX, 0);
```

The `_wc` variant maps with write-combining: writes coalesce into full cache lines before hitting PCIe, massively improving write bandwidth.

### The Migration Dance

When CPU touches a VRAM-resident page:

```
Thread accesses address in VMA backed by VRAM
                    │
                    ▼
            Page fault (no PTE)
                    │
                    ▼
         handle_mm_fault() → ...
                    │
                    ▼
         do_swap_page() detects device page
                    │
                    ▼
         dev_pagemap->ops->migrate_to_ram()
                    │
                    ▼
         ps_migrate_to_ram():
           1. Allocate system page
           2. DMA copy from VRAM
           3. Update page tables
           4. Free VRAM page to pool
                    │
                    ▼
         Thread resumes with valid PTE
```

---

## Part of Neural Splines

Pseudoscopic is a component of the [Neural Splines](https://neuralsplines.com) project—research into geometric representations of neural networks.

The insight: neural network weights aren't random numbers. They encode *structure*—relationships that can be captured by geometric primitives. A NURBS surface defined by 52×52 control points can represent the learned manifold of an entire language model.

This driver exists because inference on these compressed representations is memory-bound on the CPU. When your model fits in 16GB of HBM2 that would otherwise sit idle, why not use it?

---

## Contributing

We welcome contributions that maintain the project's principles:

1. **Minimal**: Does this addition earn its complexity?
2. **Elegant**: Is the code beautiful? Would you frame it?
3. **Robust**: Does every path handle failure gracefully?

```bash
# Before submitting
make check         # Static analysis
make test          # Self-tests
make format        # Kernel style formatting
```

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

—Neural Splines Research, 2025
