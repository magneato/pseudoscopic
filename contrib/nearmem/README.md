```
    ╔═══════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                   ║
    ║     ███╗   ██╗███████╗ █████╗ ██████╗ ███╗   ███╗███████╗███╗   ███╗              ║
    ║     ████╗  ██║██╔════╝██╔══██╗██╔══██╗████╗ ████║██╔════╝████╗ ████║              ║
    ║     ██╔██╗ ██║█████╗  ███████║██████╔╝██╔████╔██║█████╗  ██╔████╔██║              ║
    ║     ██║╚██╗██║██╔══╝  ██╔══██║██╔══██╗██║╚██╔╝██║██╔══╝  ██║╚██╔╝██║              ║
    ║     ██║ ╚████║███████╗██║  ██║██║  ██║██║ ╚═╝ ██║███████╗██║ ╚═╝ ██║              ║
    ║     ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝     ╚═╝              ║
    ║                                                                                   ║
    ║        ┌─────────────────────────────────────────────────────────────────┐        ║
    ║        │      Near-Memory Computing with Pseudoscopic                    │        ║
    ║        │      ─────────────────────────────────────────────              │        ║
    ║        │      Where data rests, and computation arrives                  │        ║
    ║        └─────────────────────────────────────────────────────────────────┘        ║
    ╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## ◈ The Fundamental Insight

There's a curious inversion at the heart of modern computing. We've spent decades making processors faster—from the 4004's 740 kHz to the H100's 1.8 GHz multi-billion transistor behemoths—but we've neglected the humble act of *getting data to those processors*.

Traditional GPU computing suffers from this asymmetry:

```
    ┌────────────────────────────────────────────────────────────────────────────────┐
    │                           THE TRADITIONAL DANCE                                │
    ├────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                │
    │    CPU RAM                                                         GPU VRAM   │
    │   ┌────────┐                                                     ┌────────┐   │
    │   │ 50 GB  │═══════════════════════════════════════════════════▶│ 50 GB  │   │
    │   │ Data   │         Transfer @ 12 GB/s = 4.2 seconds           │ Data   │   │
    │   └────────┘                                                     └────────┘   │
    │                                      │                                         │
    │                                      ▼                                         │
    │                              ┌──────────────┐                                  │
    │                              │    COMPUTE   │                                  │
    │                              │   ~200 ms    │                                  │
    │                              └──────────────┘                                  │
    │                                      │                                         │
    │                                      ▼                                         │
    │   ┌────────┐                                                     ┌────────┐   │
    │   │ Result │◀═══════════════════════════════════════════════════│ Result │   │
    │   └────────┘         Transfer @ 12 GB/s = variable               └────────┘   │
    │                                                                                │
    │   Total Pipeline: 4.4 seconds (Compute is 4.5% of time)                        │
    └────────────────────────────────────────────────────────────────────────────────┘
```

For a 50 GB dataset, you spend **4+ seconds** moving bytes across PCIe, then a measly 200 milliseconds actually processing. The GPU—a thermonuclear computational engine—sits idle 95% of the time, waiting for data.

> **Fun fact**: The PCIe 3.0 x16 link (12.8 GB/s theoretical, ~12 GB/s practical) has roughly the bandwidth of DDR2-400 RAM from 2004. We're feeding 2024 GPUs through a memory straw from two decades ago.

---

## ◈ The Inversion

**Near-Memory Computing inverts this relationship.** Instead of moving data to computation, we bring computation to data.

```
    ┌────────────────────────────────────────────────────────────────────────────────┐
    │                           THE NEAR-MEMORY WAY                                  │
    ├────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                │
    │                               GPU VRAM (via /dev/psdisk0)                      │
    │                              ┌─────────────────────────────┐                   │
    │                              │                             │                   │
    │   CPU writes via mmap()  ──▶│         DATA LIVES HERE     │◀── GPU accesses  │
    │   (directly to VRAM)        │                             │    natively       │
    │                              │    Same physical memory     │                   │
    │                              │    Two access paths          │                   │
    │                              │    Zero round-trips          │                   │
    │                              │                             │                   │
    │                              └─────────────────────────────┘                   │
    │                                          │                                     │
    │                                          ▼                                     │
    │                                  ┌──────────────┐                              │
    │                                  │    COMPUTE   │                              │
    │                                  │   In-place   │                              │
    │                                  │   ~200 ms    │                              │
    │                                  └──────────────┘                              │
    │                                          │                                     │
    │                               CPU reads results (sparse)                       │
    │                                                                                │
    │     Total Pipeline: 0.2 seconds (Compute is 100% of time)                      │
    └────────────────────────────────────────────────────────────────────────────────┘
```

The GPU becomes a **memory-side accelerator**—a coprocessor that transforms data in-place while the CPU orchestrates the show.

---

## ◈ The Physical Reality

How does one pointer address both CPU and GPU memory? The answer lies in a hardware feature that's been hiding in plain sight for years:

```
    ┌─────────────────────────────────────────────────────────────────────────────────┐
    │                              GPU Card (Physical)                                │
    ├─────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                 │
    │      ┌─────────────────────────────────────────────────────────────────┐        │
    │      │                         VRAM                                    │        │
    │      │   ┌───────────────────────────────────────────────────────┐    │        │
    │      │   │                                                       │    │        │
    │      │   │        Your Data (50 GB of glorious bytes)            │    │        │
    │      │   │                                                       │    │        │
    │      │   │   GPU Internal Access: HBM2 @ 700 GB/s                │    │        │
    │      │   │   (Native fabric, no PCIe involved)                   │    │        │
    │      │   │                                                       │    │        │
    │      │   └───────────────────────────────────────────────────────┘    │        │
    │      │                              │                                  │        │
    │      │                              │ BAR1 Aperture                   │        │
    │      │                              │ (PCIe Window)                   │        │
    │      │                              │                                  │        │
    │      └──────────────────────────────┼──────────────────────────────────┘        │
    │                                     │                                           │
    │ ════════════════════════════════════╪══════════════════════════════════════    │
    │                              PCIe Bus                                           │
    │ ════════════════════════════════════╪══════════════════════════════════════    │
    │                                     │                                           │
    │                              ┌──────┴──────┐                                    │
    │                              │     CPU     │                                    │
    │                              │   mmap()    │                                    │
    │                              │             │                                    │
    │                              │  Sees same  │                                    │
    │                              │  bytes via  │                                    │
    │                              │  BAR1 addr  │                                    │
    │                              └─────────────┘                                    │
    └─────────────────────────────────────────────────────────────────────────────────┘
```

The **BAR1 aperture** (Base Address Register 1) is a window through which the CPU can directly access GPU VRAM. In the old days (pre-2016), this window was limited—256 MB or less. Modern GPUs with **Large BAR** expose their *entire* VRAM.

Pseudoscopic exploits this: it presents a block device (`/dev/psdisk0`) backed by BAR1. When you `mmap()` that device, you get a CPU pointer to GPU memory.

> **Historical aside**: The PCI specification has supported BARs since 1993. What changed is *size*—32-bit address limitations kept BARs small until "Above 4G Decoding" became common in UEFI. Now we can expose 80+ GB through a single BAR.

---

## ◈ The API at a Glance

### Initialization

```c
// Explicit device selection
nearmem_ctx_t ctx;
nearmem_init(&ctx, "/dev/psdisk0", 0);  // GPU 0

// Auto-detection (finds first pseudoscopic device)
nearmem_init_auto(&ctx);
```

### Memory Allocation

```c
// Allocate a shared region (accessible by CPU and GPU)
nearmem_region_t region;
nearmem_alloc(&ctx, &region, 16ULL << 30);  // 16 GB

// After this:
//   region.cpu_ptr → CPU-visible pointer
//   region.gpu_ptr → GPU device pointer (offset)
//   Both point to the same physical VRAM
```

### The Synchronization Ritual

CPU and GPU have different memory models. Writes may be buffered, cached, reordered. Synchronization makes visibility explicit:

```c
// CPU has written data; tell GPU it's ready
nearmem_sync(&ctx, NEARMEM_SYNC_CPU_TO_GPU);

// GPU has computed; tell CPU results are visible
nearmem_sync(&ctx, NEARMEM_SYNC_GPU_TO_CPU);

// Full bidirectional barrier
nearmem_sync(&ctx, NEARMEM_SYNC_FULL);
```

> **Why explicit sync?** Implicit synchronization (like Unified Memory's page faults) sounds nice but hides costs. When GPU work takes 3× longer than expected, you want to know *why*. Explicit sync is instrumentable, debuggable, predictable.

### Built-in Operations

```c
// Memory manipulation
nearmem_memset(&ctx, &region, 0x00, 0, size);
nearmem_memcpy(&ctx, &dst, dst_off, &src, src_off, size);

// Pattern search (grep-like)
nearmem_find(&ctx, &region, "ERROR", 5, &offset);
nearmem_count_matches(&ctx, &region, "WARNING", 7, &count);

// Byte transform (apply 256-entry LUT)
uint8_t to_upper[256];  // Populate with uppercase mapping
nearmem_transform(&ctx, &region, to_upper);

// Analysis
uint64_t histogram[256];
nearmem_histogram(&ctx, &region, histogram);

// Reduction
float sum;
nearmem_reduce_sum_f32(&ctx, &region, float_count, &sum);

// Sorting
nearmem_sort_u32(&ctx, &region, element_count);
```

---

## ◈ Performance Characteristics

### The Honest Numbers

| Operation | Dataset | PCIe Traffic | GPU Time | Traditional Equivalent |
|-----------|---------|--------------|----------|------------------------|
| Histogram | 50 GB | 2 KB (result) | ~80 ms | 4.3s (copy+compute+copy) |
| Pattern Search | 50 GB | ~1 KB (offsets) | ~100 ms | 4.3s |
| Byte Transform | 24 GB | 0 (in-place) | ~50 ms | 4.1s |
| Sort (uint32) | 10 GB | 0 (in-place) | ~500 ms | 1.5s |
| Sum Reduction | 10 GB | 8 bytes | ~20 ms | 1.3s |

The pattern is clear: when your operation produces *small output from large input*, near-memory wins dramatically. When output size approaches input size, the advantage shrinks.

### When Near-Memory Excels

✅ **Ideal workloads:**
- Log analysis at scale (grep, histogram, regex)
- Data validation (checksums, format verification)
- In-place transformations (encoding, compression, encryption)
- Large dataset filtering (SELECT WHERE)
- Bulk sorting and deduplication
- Streaming ingestion with continuous processing

❌ **Less ideal:**
- Iterative algorithms needing CPU feedback each iteration
- Workloads where output size ≈ input size
- Small datasets where PCIe overhead is negligible
- Random access patterns (PCIe latency doesn't amortize)

---

## ◈ Practical Examples

### 1. Log Processing Pipeline

```c
// Stream logs directly to VRAM (bypasses CPU RAM entirely)
system("dd if=access.log of=/dev/psdisk0 bs=1M");

nearmem_ctx_t ctx;
nearmem_init_auto(&ctx);

// Map the log region
nearmem_region_t logs;
nearmem_map_offset(&ctx, &logs, 0, log_size);

// GPU-accelerated grep
int64_t first_error;
nearmem_find(&ctx, &logs, "ERROR", 5, &first_error);

// GPU-accelerated statistics
uint64_t byte_histogram[256];
nearmem_histogram(&ctx, &logs, byte_histogram);

// Only transfer results (bytes, not gigabytes)
printf("First error at byte %ld\n", first_error);
printf("Space characters: %lu\n", byte_histogram[' ']);
```

### 2. LLM KV-Cache Spillover

```c
// When CPU RAM fills, KV-cache spills to VRAM
nearmem_region_t kv_overflow;
nearmem_alloc(&ctx, &kv_overflow, 32ULL << 30);  // 32 GB

// Attention computation accesses spilled cache in-place
// No cudaMemcpy dance—just pointer math
void *kv_ptr = (char*)kv_overflow.cpu_ptr + layer_offset;
```

### 3. Database Cold Buffer

```c
// Cold pages live in VRAM (16 GB on this GPU)
nearmem_region_t cold_buffer;
nearmem_alloc(&ctx, &cold_buffer, 16ULL << 30);

// GPU can scan cold pages at 700 GB/s
int64_t key_offset;
nearmem_find(&ctx, &cold_buffer, key_bytes, key_len, &key_offset);

// Only hot rows come to CPU
if (key_offset >= 0) {
    nearmem_sync(&ctx, NEARMEM_SYNC_GPU_TO_CPU);
    Row *row = (Row*)((char*)cold_buffer.cpu_ptr + key_offset);
    process_row(row);
}
```

---

## ◈ The Synchronization Dance (Detailed)

For those who want to understand *why* synchronization is necessary:

```
    CPU writes to region.cpu_ptr
            │
            ▼
    ┌───────────────────┐
    │  Write-Combine    │  ← CPU writes go into WC buffers (64 bytes each)
    │     Buffers       │     These are NOT in VRAM yet!
    └───────┬───────────┘
            │
        sfence + msync         ← NEARMEM_SYNC_CPU_TO_GPU
            │                    Forces WC buffer flush
            ▼
    ┌───────────────────┐
    │      VRAM         │  ← Data now visible to GPU SM threads
    └───────┬───────────┘
            │
       GPU kernel runs
            │
            ▼
    ┌───────────────────┐
    │   GPU L2 Cache    │  ← Results may be cached in GPU L2
    └───────┬───────────┘
            │
       cudaDeviceSynchronize   ← NEARMEM_SYNC_GPU_TO_CPU
            │                    Wait for GPU, flush L2
            ▼
    ┌───────────────────┐
    │      VRAM         │  ← Results now in VRAM (not L2)
    └───────────────────┘
            │
       CPU reads via mmap (uncached access through BAR1)
```

> **The x86 memory model footnote**: On x86-64, stores to write-combining memory (including BAR1) go into WC buffers and may be coalesced/reordered. `sfence` ensures prior stores are globally visible. `msync(MS_SYNC)` forces buffer flush at kernel level. Without these, GPU may see stale data.

---

## ◈ Building Near-Memory

### Prerequisites

- **Pseudoscopic driver** loaded (see main project README)
- **CUDA Toolkit** 11.0+ (for GPU kernel support)
- **GCC** 10+ (or your kernel's compiler version)

### Compilation

```bash
cd contrib/nearmem

# Standard build (auto-detects CUDA)
make

# Explicit CUDA path
make CUDA_PATH=/usr/local/cuda-12.3

# Debug build
make DEBUG=1

# Check CUDA detection
make cuda-info
```

### Verification

```bash
# Ensure pseudoscopic is loaded
lsmod | grep pseudoscopic

# Ensure device exists
ls -la /dev/psdisk*

# Run a quick test
./log_analyzer /dev/psdisk0 "test"
```

---

## ◈ Architectural Comparison

| Approach | Data Copy | GPU Acceleration | Transparency | Predictability |
|----------|-----------|------------------|--------------|----------------|
| Standard CUDA | 2× PCIe | ✓ | Low | High |
| Unified Memory | Auto | ✓ | High | Low |
| **Near-Memory** | **0** | ✓ | Medium | High |
| NVMe Direct | 0 | ✗ | Low | High |

Near-memory is the sweet spot when you need:
- GPU acceleration (unlike NVMe direct)
- Predictable performance (unlike Unified Memory)
- Minimal data movement (unlike standard CUDA)

---

## ◈ Limitations (Honest Assessment)

1. **Initial load still crosses PCIe** — Near-memory optimizes *repeated* operations, not single passes
2. **Random CPU reads are slow** — BAR1 access is ~100ns per 64B; don't iterate byte-by-byte
3. **GPU memory is finite** — 16-80 GB depending on GPU model
4. **Requires pseudoscopic driver** — Not portable without the kernel module
5. **Write-combining semantics** — CPU writes may reorder; explicit sync required

---

## ◈ Future Horizons

- [ ] AVX-512 optimized CPU fallback paths
- [ ] Async streaming API with double-buffering
- [ ] Multi-GPU orchestration (cross-GPU copy)
- [ ] Python bindings with NumPy interop
- [ ] Apache Arrow integration
- [ ] RDMA support for distributed near-memory

---

```
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║      "The fastest data transfer is the one that never happens.                ║
    ║       The best optimization is avoiding the operation entirely."              ║
    ║                                                                               ║
    ║                                        — Neural Splines Research, 2026        ║
    ║                                           Asymmetric Solutions                ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## ◈ License

MIT License — Use freely, modify boldly, attribute kindly.

Copyright © 2026 Neural Splines LLC

---

*Data doesn't move. Computation arrives.* 🍪
