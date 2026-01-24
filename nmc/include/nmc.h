/* SPDX-License-Identifier: MIT */
/*
 * nmc.h - Near-Memory Computing Library
 *
 * Asymmetric solutions: Data stays in VRAM. Computation comes to it.
 *
 * The Traditional Model (wasteful):
 *   1. Data in CPU RAM
 *   2. cudaMemcpy to GPU (PCIe transfer #1)
 *   3. GPU processes
 *   4. cudaMemcpy back (PCIe transfer #2)
 *
 * The NMC Model (efficient):
 *   1. Data loaded directly to VRAM (PCIe transfer #1, ONLY)
 *   2. CPU reads metadata/control via pseudoscopic
 *   3. GPU transforms data in-place (700 GB/s internal bandwidth)
 *   4. CPU reads results on-demand (sparse, only what's needed)
 *
 * When 50 GB of data only needs 1 MB of results extracted,
 * NMC wins by 100x.
 *
 * Copyright (C) 2025 Neural Splines LLC
 * Author: Robert L. Sitton, Jr.
 */

#ifndef _NMC_H_
#define _NMC_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Version
 */
#define NMC_VERSION_MAJOR   0
#define NMC_VERSION_MINOR   1
#define NMC_VERSION_PATCH   0
#define NMC_VERSION_STRING  "0.1.0"

/*
 * Error Codes
 */
typedef enum {
    NMC_SUCCESS = 0,
    NMC_ERROR_NO_DEVICE,        /* Pseudoscopic device not found */
    NMC_ERROR_NO_CUDA,          /* CUDA runtime not available */
    NMC_ERROR_MMAP_FAILED,      /* Failed to mmap VRAM region */
    NMC_ERROR_CUDA_FAILED,      /* CUDA operation failed */
    NMC_ERROR_OUT_OF_MEMORY,    /* VRAM exhausted */
    NMC_ERROR_INVALID_ARG,      /* Invalid argument */
    NMC_ERROR_SYNC_TIMEOUT,     /* Synchronization timeout */
    NMC_ERROR_NOT_ALIGNED,      /* Address/size not properly aligned */
    NMC_ERROR_BUSY,             /* Resource currently in use */
} nmc_error_t;

/*
 * Memory Flags
 */
typedef enum {
    NMC_MEM_READ       = 0x01,  /* CPU can read */
    NMC_MEM_WRITE      = 0x02,  /* CPU can write */
    NMC_MEM_GPU_ONLY   = 0x04,  /* Hint: GPU-heavy access pattern */
    NMC_MEM_CPU_ONLY   = 0x08,  /* Hint: CPU-heavy access pattern */
    NMC_MEM_SHARED     = 0x10,  /* Frequent CPU-GPU alternation */
    NMC_MEM_PERSISTENT = 0x20,  /* Keep mapped across operations */
} nmc_mem_flags_t;

/*
 * Synchronization Modes
 */
typedef enum {
    NMC_SYNC_NONE,              /* No sync (caller manages) */
    NMC_SYNC_CPU_TO_GPU,        /* Ensure CPU writes visible to GPU */
    NMC_SYNC_GPU_TO_CPU,        /* Ensure GPU writes visible to CPU */
    NMC_SYNC_FULL,              /* Full bidirectional barrier */
} nmc_sync_mode_t;

/*
 * Forward Declarations
 */
typedef struct nmc_context nmc_context_t;
typedef struct nmc_region nmc_region_t;
typedef struct nmc_stream nmc_stream_t;

/*
 * Statistics
 */
typedef struct {
    uint64_t bytes_to_vram;     /* Total bytes written to VRAM */
    uint64_t bytes_from_vram;   /* Total bytes read from VRAM */
    uint64_t gpu_ops;           /* GPU operations executed */
    uint64_t cpu_touches;       /* CPU accesses to VRAM */
    uint64_t sync_count;        /* Synchronization barriers */
    uint64_t pcie_avoided;      /* Estimated bytes NOT transferred */
    double   time_in_gpu_ms;    /* Time spent in GPU kernels */
    double   time_in_sync_ms;   /* Time spent synchronizing */
} nmc_stats_t;

/*
 * =============================================================
 * Context Management
 * =============================================================
 */

/*
 * nmc_init - Initialize NMC library
 * @device_path: Path to pseudoscopic device (NULL for auto-detect)
 * @cuda_device: CUDA device index (-1 for auto-match)
 * @ctx_out:     Output context pointer
 *
 * Creates an NMC context bound to a pseudoscopic device and
 * matching CUDA device. The devices must refer to the same
 * physical GPU.
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_init(const char *device_path, 
                     int cuda_device,
                     nmc_context_t **ctx_out);

/*
 * nmc_shutdown - Destroy NMC context
 * @ctx: Context to destroy
 *
 * Releases all resources. Outstanding regions become invalid.
 */
void nmc_shutdown(nmc_context_t *ctx);

/*
 * nmc_get_stats - Get performance statistics
 * @ctx:   Context
 * @stats: Output statistics structure
 */
nmc_error_t nmc_get_stats(nmc_context_t *ctx, nmc_stats_t *stats);

/*
 * nmc_reset_stats - Reset statistics counters
 * @ctx: Context
 */
void nmc_reset_stats(nmc_context_t *ctx);

/*
 * nmc_get_capacity - Get total VRAM capacity
 * @ctx: Context
 *
 * Returns: Total bytes of VRAM available
 */
size_t nmc_get_capacity(nmc_context_t *ctx);

/*
 * nmc_get_available - Get available VRAM
 * @ctx: Context
 *
 * Returns: Bytes of VRAM not currently allocated
 */
size_t nmc_get_available(nmc_context_t *ctx);

/*
 * =============================================================
 * Memory Region Management
 * =============================================================
 */

/*
 * nmc_alloc - Allocate a VRAM region
 * @ctx:    Context
 * @size:   Size in bytes (will be rounded up to page boundary)
 * @flags:  Memory flags
 * @region: Output region pointer
 *
 * Allocates a contiguous region of VRAM accessible by both
 * CPU and GPU. The region is not initialized.
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_alloc(nmc_context_t *ctx,
                      size_t size,
                      nmc_mem_flags_t flags,
                      nmc_region_t **region);

/*
 * nmc_free - Free a VRAM region
 * @region: Region to free
 *
 * Releases the region. Pointers become invalid.
 */
void nmc_free(nmc_region_t *region);

/*
 * nmc_cpu_ptr - Get CPU-accessible pointer to region
 * @region: Region
 *
 * Returns write-combining mapped pointer. Suitable for
 * bulk sequential writes. Random reads are slow.
 *
 * Returns: Pointer or NULL if not CPU-accessible
 */
void *nmc_cpu_ptr(nmc_region_t *region);

/*
 * nmc_gpu_ptr - Get GPU-accessible pointer to region
 * @region: Region
 *
 * Returns device pointer usable in CUDA kernels.
 *
 * Returns: Device pointer or NULL
 */
void *nmc_gpu_ptr(nmc_region_t *region);

/*
 * nmc_size - Get region size
 * @region: Region
 *
 * Returns: Size in bytes
 */
size_t nmc_size(nmc_region_t *region);

/*
 * nmc_load - Load data into VRAM region
 * @region: Destination region
 * @offset: Offset within region
 * @src:    Source buffer in system RAM
 * @size:   Bytes to copy
 *
 * Efficiently loads data using write-combining.
 * This is the ONE time data crosses PCIe.
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_load(nmc_region_t *region,
                     size_t offset,
                     const void *src,
                     size_t size);

/*
 * nmc_load_file - Load file directly into VRAM
 * @region: Destination region
 * @offset: Offset within region
 * @fd:     File descriptor (must support mmap)
 * @file_offset: Offset within file
 * @size:   Bytes to copy
 *
 * Streams file content to VRAM without intermediate copy.
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_load_file(nmc_region_t *region,
                          size_t offset,
                          int fd,
                          size_t file_offset,
                          size_t size);

/*
 * nmc_extract - Extract data from VRAM region
 * @region: Source region
 * @offset: Offset within region
 * @dst:    Destination buffer in system RAM
 * @size:   Bytes to copy
 *
 * Reads data from VRAM. Use sparingly - this is
 * where PCIe becomes a bottleneck.
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_extract(nmc_region_t *region,
                        size_t offset,
                        void *dst,
                        size_t size);

/*
 * =============================================================
 * Synchronization
 * =============================================================
 */

/*
 * nmc_sync - Synchronize CPU and GPU views
 * @ctx:  Context
 * @mode: Synchronization mode
 *
 * Ensures memory consistency between CPU and GPU.
 *
 * CPU_TO_GPU: After CPU writes, before GPU reads
 * GPU_TO_CPU: After GPU writes, before CPU reads
 * FULL: Both directions (expensive)
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_sync(nmc_context_t *ctx, nmc_sync_mode_t mode);

/*
 * nmc_fence_cpu - CPU-side memory fence
 * @region: Region to fence (NULL for all)
 *
 * Ensures all CPU writes to the region are visible.
 * Lighter weight than full sync.
 */
void nmc_fence_cpu(nmc_region_t *region);

/*
 * nmc_fence_gpu - GPU-side memory fence
 * @ctx: Context
 *
 * Ensures all GPU operations complete.
 */
nmc_error_t nmc_fence_gpu(nmc_context_t *ctx);

/*
 * =============================================================
 * Stream Management (Async Operations)
 * =============================================================
 */

/*
 * nmc_stream_create - Create an async operation stream
 * @ctx:    Context
 * @stream: Output stream pointer
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_stream_create(nmc_context_t *ctx, nmc_stream_t **stream);

/*
 * nmc_stream_destroy - Destroy a stream
 * @stream: Stream to destroy
 */
void nmc_stream_destroy(nmc_stream_t *stream);

/*
 * nmc_stream_sync - Wait for stream operations to complete
 * @stream: Stream to synchronize
 *
 * Returns: NMC_SUCCESS or error code
 */
nmc_error_t nmc_stream_sync(nmc_stream_t *stream);

/*
 * =============================================================
 * Built-in GPU Operations (In-Place Transforms)
 * =============================================================
 * These operate on data IN VRAM without crossing PCIe.
 */

/*
 * nmc_memset - Fill region with byte value
 * @region: Target region
 * @offset: Offset within region
 * @value:  Byte value to fill
 * @size:   Bytes to fill
 * @stream: Async stream (NULL for sync)
 *
 * GPU-accelerated memset. ~700 GB/s on modern GPUs.
 */
nmc_error_t nmc_memset(nmc_region_t *region,
                       size_t offset,
                       int value,
                       size_t size,
                       nmc_stream_t *stream);

/*
 * nmc_memcpy_internal - Copy within VRAM
 * @dst:        Destination region
 * @dst_offset: Offset in destination
 * @src:        Source region
 * @src_offset: Offset in source
 * @size:       Bytes to copy
 * @stream:     Async stream (NULL for sync)
 *
 * GPU-accelerated copy, entirely within VRAM.
 * Does NOT cross PCIe.
 */
nmc_error_t nmc_memcpy_internal(nmc_region_t *dst,
                                size_t dst_offset,
                                nmc_region_t *src,
                                size_t src_offset,
                                size_t size,
                                nmc_stream_t *stream);

/*
 * nmc_transform_u8 - Apply byte-wise transform
 * @region: Target region
 * @offset: Offset within region
 * @size:   Bytes to transform
 * @table:  256-byte lookup table (will be copied to GPU)
 * @stream: Async stream (NULL for sync)
 *
 * Applies LUT transform to each byte. Useful for:
 * - Case conversion
 * - Character class detection
 * - Simple encryption/encoding
 */
nmc_error_t nmc_transform_u8(nmc_region_t *region,
                             size_t offset,
                             size_t size,
                             const uint8_t table[256],
                             nmc_stream_t *stream);

/*
 * nmc_search - Search for pattern in region
 * @region:      Region to search
 * @offset:      Start offset
 * @size:        Bytes to search
 * @pattern:     Pattern to find
 * @pattern_len: Length of pattern
 * @results:     Output region for match offsets (uint64_t array)
 * @max_results: Maximum matches to record
 * @count_out:   Output: actual match count
 * @stream:      Async stream (NULL for sync)
 *
 * GPU-parallel pattern search. Results stay in VRAM.
 * CPU only reads the count and specific matches of interest.
 */
nmc_error_t nmc_search(nmc_region_t *region,
                       size_t offset,
                       size_t size,
                       const void *pattern,
                       size_t pattern_len,
                       nmc_region_t *results,
                       size_t max_results,
                       uint64_t *count_out,
                       nmc_stream_t *stream);

/*
 * nmc_reduce_sum_f32 - Sum float array
 * @region:  Region containing float array
 * @offset:  Offset within region
 * @count:   Number of floats
 * @sum_out: Output sum
 * @stream:  Async stream (NULL for sync)
 *
 * GPU-parallel reduction. Only the scalar result crosses PCIe.
 */
nmc_error_t nmc_reduce_sum_f32(nmc_region_t *region,
                               size_t offset,
                               size_t count,
                               float *sum_out,
                               nmc_stream_t *stream);

/*
 * nmc_reduce_max_f32 - Find maximum in float array
 * @region:  Region containing float array
 * @offset:  Offset within region
 * @count:   Number of floats
 * @max_out: Output maximum
 * @idx_out: Output index of maximum (NULL to skip)
 * @stream:  Async stream (NULL for sync)
 */
nmc_error_t nmc_reduce_max_f32(nmc_region_t *region,
                               size_t offset,
                               size_t count,
                               float *max_out,
                               uint64_t *idx_out,
                               nmc_stream_t *stream);

/*
 * nmc_histogram_u8 - Compute histogram of byte values
 * @region:     Region containing byte data
 * @offset:     Offset within region
 * @size:       Bytes to histogram
 * @hist_out:   Output: 256-element histogram (in system RAM)
 * @stream:     Async stream (NULL for sync)
 *
 * Computes histogram entirely in GPU, only 256 values cross PCIe.
 */
nmc_error_t nmc_histogram_u8(nmc_region_t *region,
                             size_t offset,
                             size_t size,
                             uint64_t hist_out[256],
                             nmc_stream_t *stream);

/*
 * nmc_sort_u32 - Sort 32-bit unsigned integers in-place
 * @region: Region containing uint32_t array
 * @offset: Offset within region
 * @count:  Number of elements
 * @stream: Async stream (NULL for sync)
 *
 * GPU radix sort. Data never leaves VRAM.
 */
nmc_error_t nmc_sort_u32(nmc_region_t *region,
                         size_t offset,
                         size_t count,
                         nmc_stream_t *stream);

/*
 * nmc_sort_f32 - Sort 32-bit floats in-place
 * @region: Region containing float array
 * @offset: Offset within region
 * @count:  Number of elements
 * @stream: Async stream (NULL for sync)
 */
nmc_error_t nmc_sort_f32(nmc_region_t *region,
                         size_t offset,
                         size_t count,
                         nmc_stream_t *stream);

/*
 * =============================================================
 * Custom Kernel Support
 * =============================================================
 */

/*
 * nmc_launch_kernel - Launch custom CUDA kernel on region
 * @ctx:        Context
 * @kernel:     CUDA kernel function pointer
 * @grid:       Grid dimensions
 * @block:      Block dimensions
 * @shared_mem: Shared memory size
 * @stream:     Async stream (NULL for sync)
 * @...:        Kernel arguments (must include nmc_gpu_ptr results)
 *
 * For advanced users who need custom GPU operations.
 * The kernel operates on VRAM data in-place.
 */
nmc_error_t nmc_launch_kernel(nmc_context_t *ctx,
                              void *kernel,
                              dim3 grid,
                              dim3 block,
                              size_t shared_mem,
                              nmc_stream_t *stream,
                              void **args);

/*
 * =============================================================
 * Utility Functions
 * =============================================================
 */

/*
 * nmc_error_string - Get human-readable error description
 * @error: Error code
 *
 * Returns: Static string describing error
 */
const char *nmc_error_string(nmc_error_t error);

/*
 * nmc_print_stats - Print statistics to file
 * @ctx: Context
 * @fp:  Output file (stderr if NULL)
 */
void nmc_print_stats(nmc_context_t *ctx, FILE *fp);

#ifdef __cplusplus
}
#endif

#endif /* _NMC_H_ */
