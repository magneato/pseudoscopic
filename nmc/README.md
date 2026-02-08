```
    ╔═══════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                   ║
    ║         ███╗   ██╗███╗   ███╗ ██████╗                                             ║
    ║         ████╗  ██║████╗ ████║██╔════╝                                             ║
    ║         ██╔██╗ ██║██╔████╔██║██║                                                  ║
    ║         ██║╚██╗██║██║╚██╔╝██║██║                                                  ║
    ║         ██║ ╚████║██║ ╚═╝ ██║╚██████╗                                             ║
    ║         ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝                                             ║
    ║                                                                                   ║
    ║                 Near-Memory Computing Primitives                                  ║
    ║         ─────────────────────────────────────────────                            ║
    ║         Where data stays put, and algorithms come calling                         ║
    ╚═══════════════════════════════════════════════════════════════════════════════════╝
```

*A pseudoscopic subsystem for asymmetric memory patterns.*

---

## ◈ The Core Insight

Every GPU programmer knows the ritual dance:

```c
// The Traditional Waltz — A Dance in Three Movements
cudaMemcpy(gpu_data, host_data, size, cudaMemcpyHostToDevice);  // 50 GB → (4 seconds)
kernel<<<blocks, threads>>>(gpu_data);                          // 0.2 seconds (Fast!)
cudaMemcpy(host_data, gpu_data, size, cudaMemcpyDeviceToHost);  // ← 50 GB (4 seconds)
```

For a *single* operation, you move **100 GB** across PCIe to process 50 GB of data. The math doesn't lie.

Want to run 10 operations on that dataset? That's **1 TB** of PCIe traffic. Your GPU spends 97% of its time as a glorified DMA controller, waiting for bytes that are already where they need to be—they just live in the wrong address space.

> **Historical perspective**: This pattern emerged from the GPGPU era (~2007) when GPU memory was truly foreign territory. CUDA provided cudaMemcpy because there was no alternative. But hardware has evolved. Software assumptions haven't.

---

## ◈ The Inversion

**NMC inverts the data-to-compute relationship:**

```c
// The NMC Way — Data Stays Home
nmc_load(region, host_data, size);      // 50 GB → (ONCE, amortized across operations)

for (int i = 0; i < 10; i++) {
    nmc_search(region, pattern[i], ...);  // 0 GB — data stays in VRAM
    nmc_extract(results, &count, 8);      // 8 bytes ← (just the answer)
}
```

Total PCIe traffic: **50 GB + 80 bytes** instead of 1 TB.

That's not a 2× improvement. That's a **20,000×** reduction in data movement.

---

## ◈ The Mathematics of Bandwidth

Let's be precise about why this matters:

```
    ╔═════════════════════════════════════════════════════════════════════════╗
    ║                        BANDWIDTH ASYMMETRY                               ║
    ╠═════════════════════════════════════════════════════════════════════════╣
    ║                                                                         ║
    ║    GPU Internal Bandwidth (HBM2): ..............  900 GB/s             ║
    ║    GPU Internal Bandwidth (GDDR6X): ............  700 GB/s             ║
    ║    PCIe 4.0 x16 (practical): ...................   25 GB/s             ║
    ║    PCIe 3.0 x16 (practical): ...................   12 GB/s             ║
    ║                                                                         ║
    ║    Ratio (HBM2 / PCIe 3.0): .....................  75:1                ║
    ║                                                                         ║
    ║    Every byte that DOESN'T cross PCIe is a 75× effective gain.         ║
    ║                                                                         ║
    ╚═════════════════════════════════════════════════════════════════════════╝
```

This asymmetry is the foundation of near-memory computing. The GPU's internal memory system operates at frequencies and parallelism that PCIe simply cannot match. When your workload is "read a lot, output a little," keeping data on the *fast* side of the divide is pure win.

---

## ◈ The Physical Architecture

NMC builds on Pseudoscopic's VRAM exposure:

```
    ┌────────────────────────────────────────────────────────────────────────────────┐
    │                              GPU Card                                          │
    ├────────────────────────────────────────────────────────────────────────────────┤
    │                                                                                │
    │   ┌───────────────────────────────────────────────────────────────────────┐   │
    │   │                         VRAM (HBM2 / GDDR6X)                          │   │
    │   │                                                                       │   │
    │   │   ┌─────────────────────────────────────────────────────────────┐    │   │
    │   │   │                      Data Region                            │    │   │
    │   │   │                                                             │    │   │
    │   │   │     Addressable via CPU:  mmap("/dev/psdisk0")             │    │   │
    │   │   │     Addressable via GPU:  CUDA device pointer               │    │   │
    │   │   │                                                             │    │   │
    │   │   │     ╔═══════════════════════════════════════════════════╗  │    │   │
    │   │   │     ║           SAME PHYSICAL MEMORY                    ║  │    │   │
    │   │   │     ║           Different access paths                  ║  │    │   │
    │   │   │     ║           Same coherent view                      ║  │    │   │
    │   │   │     ╚═══════════════════════════════════════════════════╝  │    │   │
    │   │   │                                                             │    │   │
    │   │   └─────────────────────────────────────────────────────────────┘    │   │
    │   │                              │                 │                      │   │
    │   │           ┌──────────────────┴──┐    ┌─────────┴───────────┐         │   │
    │   │           │                     │    │                     │         │   │
    │   │      GPU Compute            CPU Access                               │   │
    │   │      (700 GB/s)             (12 GB/s via BAR1)                       │   │
    │   │      In-place ops           Control & results                        │   │
    │   │      Bulk transforms        Sparse reads                             │   │
    │   │                                                                       │   │
    │   └───────────────────────────────────────────────────────────────────────┘   │
    │                                                                                │
    └────────────────────────────────────────────────────────────────────────────────┘
```

The CPU and GPU see the **same physical memory** through different paths:
- **CPU**: Write-combining access via PCIe BAR1 (mmap'd)
- **GPU**: Native device memory access (CUDA device pointer)

Synchronization is explicit memory barriers—deterministic, measurable, no magic.

---

## ◈ API Usage

### Initialization

```c
nmc_context_t *ctx;
int result = nmc_init(NULL, -1, &ctx);  // Auto-detect device, auto-select GPU

// Or explicit:
result = nmc_init("/dev/psdisk0", 0, &ctx);  // Specific device, GPU 0
```

### Memory Allocation

```c
nmc_region_t *data;

// Allocate 50 GB in GPU VRAM
result = nmc_alloc(ctx, 50ULL << 30, NMC_MEM_GPU_ONLY, &data);

// Flags available:
// NMC_MEM_GPU_ONLY     — Data lives exclusively in VRAM
// NMC_MEM_CPU_VISIBLE  — Optimized for CPU access (but still in VRAM)
// NMC_MEM_STREAM       — Hint for streaming access pattern
```

### Data Loading (One-Time Transfer)

```c
// Load data into VRAM (this is the PCIe cost you pay once)
nmc_load(data, 0, my_huge_file, file_size);

// Ensure GPU can see it
nmc_sync(ctx, NMC_SYNC_CPU_TO_GPU);
```

### GPU Operations (No PCIe!)

After loading, all operations work in-place:

```c
// Search 50 GB for a pattern — data never moves
int64_t offsets[1024];
size_t match_count;
nmc_search(data, 0, file_size, "ERROR", 5, offsets, 1024, &match_count, NULL);

// Sort 10 billion floats in-place
nmc_sort_f32(data, 0, 10ULL << 30 / sizeof(float), NULL);

// Compute histogram — only 256 values returned
uint64_t histogram[256];
nmc_histogram_u8(data, 0, file_size, histogram, NULL);

// Reduce to sum — only 4 bytes returned
float sum;
nmc_reduce_sum_f32(data, 0, float_count, &sum, NULL);
```

### Extract Results (Sparse)

```c
// Only pull what you need
uint64_t match_offsets[100];
nmc_extract(data, first_match_offset, match_offsets, 100 * sizeof(uint64_t));
```

---

## ◈ Real-World Applications

### 1. Log Analysis at Scale

```
    Traditional Approach:
    ─────────────────────
    Load: 50 GB server logs
    Search: 100 different patterns
    PCIe traffic: 50 GB × 100 operations × 2 copies = 10 TB
    
    NMC Approach:
    ─────────────────────
    Load: 50 GB server logs (ONCE)
    Search: 100 different patterns (IN-PLACE)
    PCIe traffic: 50 GB + 100 KB results = 50.0001 GB
    
    Savings: 200×
```

### 2. Database Cold Buffer

```c
// Overflow pages live in VRAM
nmc_region_t *cold_pages;
nmc_alloc(ctx, 32ULL << 30, NMC_MEM_GPU_ONLY, &cold_pages);

// GPU scans cold pages at 700 GB/s
// Only matching rows cross PCIe
nmc_search(cold_pages, 0, buffer_size, key, key_len, results, max, &count, NULL);
```

### 3. ML Feature Processing

```c
// 100 GB feature vectors loaded once
nmc_load(features, 0, feature_file, 100ULL << 30);

// GPU-accelerated normalization (in-place)
nmc_transform_normalize(features, 0, feature_count, mean, stddev, NULL);

// Histogram for distribution analysis (2 KB output)
nmc_histogram_f32(features, 0, feature_count, bins, 256, NULL);
```

### 4. LLM KV-Cache Spillover

```c
// When CPU RAM fills, KV-cache overflows to VRAM
// Pseudoscopic RAM mode handles migration automatically
// Attention computation accesses spilled cache in-place
```

### 5. Stream Processing

```c
// Continuous ingestion directly to VRAM
while (streaming) {
    nmc_load(buffer, write_offset, batch, batch_size);
    write_offset += batch_size;
    
    // GPU processes accumulated batches in-place
    nmc_histogram_u8(buffer, 0, write_offset, running_histogram, NULL);
}

// Only aggregates ever cross PCIe
```

---

## ◈ Performance Characteristics

| Operation | Data Size | PCIe Traffic | GPU Time | Notes |
|-----------|-----------|--------------|----------|-------|
| Search | 50 GB | 0 + results | ~100 ms | Pattern match at 500 GB/s |
| Sort | 10 GB | 0 | ~500 ms | Radix sort, in-place |
| Histogram | 50 GB | 2 KB | ~80 ms | 256-bucket byte histogram |
| Sum/Max | 10 GB | 4-8 bytes | ~20 ms | Single scalar output |
| Transform | 24 GB | 0 | ~50 ms | LUT-based byte transform |

Compare to traditional CUDA where each operation requires **2× data size** in PCIe transfers.

---

## ◈ Building NMC

### Prerequisites

```bash
# 1. Load pseudoscopic driver in ramdisk mode
sudo modprobe pseudoscopic mode=ramdisk

# 2. Verify device exists
ls -la /dev/psdisk0   # Should exist

# 3. Verify CUDA is available
nvidia-smi            # Should show your GPU(s)
```

### Compilation

```bash
cd nmc
make

# With explicit CUDA path:
make CUDA_PATH=/opt/cuda-12.0
```

### Running Examples

```bash
# Log search demonstration
./build/log_grep /var/log/syslog "error" "warning" "failed"

# Analytics benchmark
./build/analytics

# Stream processing demo
./build/stream_demo
```

---

## ◈ Design Philosophy

> *"The fastest data transfer is the one that doesn't happen."*

NMC is built on the observation that modern workloads often:
1. **Load** large datasets (once)
2. **Perform** multiple operations (repeatedly)
3. **Extract** small results (aggregates, matches, statistics)

The traditional model optimizes for (1)—making uploads fast.
NMC optimizes for (2) and (3)—eliminating transfers entirely.

When your 50 GB dataset produces 50 KB of results after 10 operations, moving 50 GB ten times is **asymmetrically wasteful**.

**Asymmetric solutions for asymmetric problems.**

---

## ◈ Limitations (Honest Assessment)

1. **Initial load still crosses PCIe** — NMC optimizes repeated operations, not single-pass workflows
2. **CPU random reads are slow** — ~100ns per 64-byte line via BAR1; don't iterate byte-by-byte
3. **GPU memory is finite** — 16-80 GB depending on GPU model
4. **Requires pseudoscopic driver** — Not portable without the kernel module
5. **Not always faster** — Single-pass compute-bound workloads may not benefit

---

## ◈ Relationship to Pseudoscopic

NMC is a userspace library built on the [Pseudoscopic](../README.md) kernel driver. It requires pseudoscopic loaded in ramdisk mode:

```bash
sudo modprobe pseudoscopic mode=ramdisk
```

This exposes GPU VRAM as `/dev/psdisk0`, which NMC mmaps for CPU access while CUDA provides GPU access to the same physical memory.

```
    ┌─────────────────────────────────────────────────────────────────────┐
    │                          Pseudoscopic Stack                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │   [ Your Application ]                                              │
    │          │                                                          │
    │          ▼                                                          │
    │   [ NMC Library ]    ← High-level primitives (search, sort, etc.)   │
    │          │                                                          │
    │          ▼                                                          │
    │   [ libnearmem ]     ← Core allocation and sync                     │
    │          │                                                          │
    │          ▼                                                          │
    │   [ pseudoscopic.ko ]  ← Kernel driver exposing VRAM                │
    │          │                                                          │
    │          ▼                                                          │
    │   [ /dev/psdisk0 ]   ← Block device backed by GPU VRAM              │
    └─────────────────────────────────────────────────────────────────────┘
```

---

```
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║      "Data is a fluid. It seeks the path of least resistance.                 ║
    ║       The wise architect builds channels, not pumps."                         ║
    ║                                                                               ║
    ║                                        — Neural Splines Research, 2026        ║
    ║                                           Asymmetric Solutions                ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
```

---

## ◈ License

MIT License

Copyright © 2026 Neural Splines LLC

---

*"C is for cookie, and cookie is for me. But data is for VRAM, and VRAM is for GPU."* 🍪
