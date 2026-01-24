# NMC - Near-Memory Computing Library

**Data stays in VRAM. Computation comes to it.**

*A pseudoscopic subsystem for asymmetric memory patterns.*

---

## The Insight

Every GPU programmer knows the dance:

```c
// The Traditional Waltz
cudaMemcpy(gpu_data, host_data, size, cudaMemcpyHostToDevice);  // 50 GB →
kernel<<<...>>>(gpu_data);                                       // Fast!
cudaMemcpy(host_data, gpu_data, size, cudaMemcpyDeviceToHost);  // ← 50 GB
```

For a single operation, you move **100 GB** across PCIe to process 50 GB of data.

Run 10 operations? That's **1 TB** of PCIe traffic.

**NMC inverts this pattern:**

```c
// The NMC Way
nmc_load(region, host_data, size);      // 50 GB → (ONCE)

for (int i = 0; i < 10; i++) {
    nmc_search(region, pattern[i], ...);  // 0 GB - stays in VRAM
    nmc_extract(results, &count, 8);      // 8 bytes ←
}
```

Total PCIe traffic: **50 GB + 80 bytes** instead of 1 TB.

---

## The Math

```
GPU Internal Bandwidth:    700 GB/s (HBM2)
PCIe 3.0 x16 Bandwidth:    12 GB/s

Ratio: 58x

Every byte we DON'T move across PCIe is a 58x effective bandwidth gain.
```

---

## How It Works

NMC builds on pseudoscopic's ramdisk mode:

```
┌────────────────────────────────────────────────────────────────┐
│                         GPU VRAM                                │
│  ┌────────────────────────────────────────────────────────┐    │
│  │                    Data Region                          │    │
│  │   Addressable by CPU (via mmap of /dev/psdisk0)        │    │
│  │   Addressable by GPU (via CUDA device pointers)        │    │
│  │   SAME PHYSICAL MEMORY                                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                              │                                  │
│         ┌────────────────────┴────────────────────┐            │
│         │                                          │            │
│    GPU Kernels                              CPU Access          │
│    (700 GB/s)                               (12 GB/s)           │
│    In-place transforms                      Control & results   │
│    Bulk operations                          Sparse reads        │
└────────────────────────────────────────────────────────────────┘
```

The CPU and GPU see the **same physical memory** through different paths:
- CPU: Write-combining mapped via PCIe BAR1
- GPU: Native device memory access

Synchronization is explicit memory barriers—no magic.

---

## API Overview

### Initialize

```c
nmc_context_t *ctx;
nmc_init(NULL, -1, &ctx);  // Auto-detect device
```

### Allocate VRAM Regions

```c
nmc_region_t *data;
nmc_alloc(ctx, 50ULL << 30, NMC_MEM_GPU_ONLY, &data);  // 50 GB
```

### Load Data (One-Time Transfer)

```c
nmc_load(data, 0, my_huge_file, file_size);
nmc_sync(ctx, NMC_SYNC_CPU_TO_GPU);
```

### GPU Operations (No PCIe!)

```c
// Search 50 GB for pattern - data never moves
nmc_search(data, 0, file_size, "ERROR", 5, results, max, &count, NULL);

// Sort 10 billion floats in-place
nmc_sort_f32(data, 0, 10ULL << 30 / sizeof(float), NULL);

// Compute histogram - only 256 values returned
nmc_histogram_u8(data, 0, file_size, histogram, NULL);
```

### Extract Results (Sparse)

```c
// Only pull what you need
nmc_extract(results, 0, &match_offsets, count * sizeof(uint64_t));
```

---

## Applications

### 1. Log Analysis
```
Load: 50 GB server logs
Search: 100 different patterns
Traditional: 50 GB × 100 × 2 = 10 TB PCIe
NMC: 50 GB + 100 KB results = 50.0001 TB PCIe
Savings: 200x
```

### 2. Database Analytics
```
Load: Buffer pool overflow pages
Scan: Complex predicates
Sort: ORDER BY on large results
All without moving data back to CPU
```

### 3. ML Feature Processing
```
Load: 100 GB feature vectors
Transform: Normalization, encoding
Histogram: Distribution analysis
Only aggregates cross PCIe
```

### 4. KV-Cache Spillover
```
LLM inference with long context
KV-cache grows beyond RAM
Spill to VRAM, migrate on-demand
Pseudoscopic RAM mode handles this transparently
```

### 5. Stream Processing
```
Continuous data ingestion to VRAM
GPU processes in batches
CPU reads aggregated results
Never buffer in CPU RAM
```

---

## Building

### Requirements
- CUDA Toolkit 11.0+
- GCC 10+
- Pseudoscopic driver in ramdisk mode

### Build

```bash
cd nmc
make
```

### Run Examples

```bash
# Load pseudoscopic in ramdisk mode first
sudo modprobe pseudoscopic mode=ramdisk

# Run log search
./build/log_grep /var/log/syslog "error" "warning" "failed"

# Run analytics benchmark
./build/analytics
```

---

## Performance Characteristics

| Operation | Data Size | PCIe Traffic | GPU Time |
|-----------|-----------|--------------|----------|
| Search | 50 GB | 0 + results | ~100 ms |
| Sort | 10 GB | 0 | ~500 ms |
| Histogram | 50 GB | 2 KB | ~80 ms |
| Sum/Max | 10 GB | 4-8 bytes | ~20 ms |

Compare to traditional approach where each operation requires 2× data size PCIe transfer.

---

## Limitations

1. **Initial load still crosses PCIe** - NMC optimizes repeated operations, not single passes
2. **Random CPU reads are slow** - Use GPU for bulk access, CPU for sparse/control
3. **GPU memory is limited** - 16-80 GB depending on GPU
4. **Requires pseudoscopic** - Not portable to systems without it

---

## Design Philosophy

> *The fastest data transfer is the one that doesn't happen.*

NMC is built on the observation that modern workloads often:
1. Load large datasets
2. Perform multiple operations
3. Extract small results

The traditional model optimizes for (1). NMC optimizes for (2) and (3).

When your 50 GB dataset produces 50 KB of results after 10 operations, moving 50 GB ten times is **asymmetrically wasteful**.

**Asymmetric solutions for asymmetric problems.**

---

## Part of Pseudoscopic

NMC is a userspace library built on the [Pseudoscopic](../README.md) kernel driver. It requires pseudoscopic loaded in ramdisk mode:

```bash
sudo modprobe pseudoscopic mode=ramdisk
```

This exposes GPU VRAM as `/dev/psdisk0`, which NMC mmaps for CPU access while CUDA provides GPU access to the same physical memory.

---

## License

MIT License

Copyright (C) 2025 Neural Splines LLC

---

*"C is for cookie, and cookie is for me. But data is for VRAM, and VRAM is for GPU."* 🍪
