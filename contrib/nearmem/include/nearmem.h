/*
 * nearmem.h - Near-Memory Computing with Pseudoscopic
 *
 * The Unconventional Insight:
 * ---------------------------
 * Data doesn't move. Computation comes to the data.
 *
 * Traditional GPU computing:
 *   CPU RAM → [PCIe copy] → GPU VRAM → [compute] → [PCIe copy] → CPU RAM
 *   Cost: 2× PCIe transfers, ~8 seconds for 50GB round-trip
 *
 * Near-Memory Computing:
 *   [VRAM] ← CPU writes via BAR1
 *   [VRAM] ← GPU computes in-place (700 GB/s internal)
 *   [VRAM] → CPU reads results via BAR1
 *   Cost: Only the bytes you touch cross PCIe
 *
 * The GPU becomes a "memory-side accelerator" - a coprocessor that
 * transforms data in-place while CPU orchestrates.
 *
 * Copyright (C) 2025 Neural Splines LLC
 * License: MIT (permissive, for maximum adoption)
 */

#ifndef _NEARMEM_H_
#define _NEARMEM_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

/*
 * Version
 */
#define NEARMEM_VERSION_MAJOR   0
#define NEARMEM_VERSION_MINOR   1
#define NEARMEM_VERSION_PATCH   0
#define NEARMEM_VERSION_STRING  "0.1.0"

/*
 * Error codes
 */
typedef enum {
    NEARMEM_OK = 0,
    NEARMEM_ERROR_INIT = -1,
    NEARMEM_ERROR_NO_DEVICE = -2,
    NEARMEM_ERROR_MMAP = -3,
    NEARMEM_ERROR_CUDA = -4,
    NEARMEM_ERROR_SYNC = -5,
    NEARMEM_ERROR_BOUNDS = -6,
    NEARMEM_ERROR_INVALID = -7,
} nearmem_error_t;

/*
 * Memory region handle
 *
 * Represents a region of VRAM that's accessible by both CPU and GPU.
 * The same physical memory, two access paths.
 */
typedef struct nearmem_region {
    void        *cpu_ptr;       /* CPU-accessible pointer (via BAR1/mmap) */
    void        *gpu_ptr;       /* GPU-accessible pointer (device memory) */
    size_t      size;           /* Region size in bytes */
    int         device_id;      /* CUDA device ID */
    int         ps_fd;          /* Pseudoscopic device fd */
    uint64_t    offset;         /* Offset within VRAM */
    bool        owned;          /* Did we allocate this? */
} nearmem_region_t;

/*
 * Context handle
 *
 * Manages the pseudoscopic device and CUDA interop state.
 */
typedef struct nearmem_ctx {
    int                 ps_fd;          /* /dev/psdiskN file descriptor */
    void                *ps_base;       /* mmap'd base address */
    size_t              ps_size;        /* Total VRAM size */
    int                 cuda_device;    /* CUDA device index */
    void                *cuda_ctx;      /* CUcontext (opaque) */
    bool                initialized;
} nearmem_ctx_t;

/*
 * Synchronization primitives
 *
 * CPU and GPU have different views of memory. These ensure coherency.
 */
typedef enum {
    NEARMEM_SYNC_CPU_TO_GPU,    /* Flush CPU writes, make visible to GPU */
    NEARMEM_SYNC_GPU_TO_CPU,    /* Wait for GPU, make results visible to CPU */
    NEARMEM_SYNC_FULL,          /* Bidirectional fence */
} nearmem_sync_t;

/*
 * Built-in kernel types
 *
 * Common operations optimized for in-place execution.
 */
typedef enum {
    /* Memory operations */
    NEARMEM_OP_MEMSET,          /* Fill with constant */
    NEARMEM_OP_MEMCPY,          /* Copy within VRAM (no PCIe) */
    NEARMEM_OP_GATHER,          /* Gather from indices */
    NEARMEM_OP_SCATTER,         /* Scatter to indices */
    
    /* Search operations */
    NEARMEM_OP_FIND_BYTE,       /* Find first occurrence of byte */
    NEARMEM_OP_FIND_PATTERN,    /* Find pattern (like grep) */
    NEARMEM_OP_COUNT_MATCHES,   /* Count pattern occurrences */
    
    /* Transform operations */
    NEARMEM_OP_TRANSFORM_U8,    /* Apply LUT to bytes */
    NEARMEM_OP_COMPRESS_RLE,    /* RLE compression */
    NEARMEM_OP_DECOMPRESS_RLE,  /* RLE decompression */
    
    /* Reduction operations */
    NEARMEM_OP_REDUCE_SUM,      /* Sum all elements */
    NEARMEM_OP_REDUCE_MIN,      /* Find minimum */
    NEARMEM_OP_REDUCE_MAX,      /* Find maximum */
    NEARMEM_OP_HISTOGRAM,       /* Compute histogram */
    
    /* Sort operations */
    NEARMEM_OP_SORT_U32,        /* Radix sort 32-bit integers */
    NEARMEM_OP_SORT_U64,        /* Radix sort 64-bit integers */
    
    /* Custom */
    NEARMEM_OP_CUSTOM,          /* User-provided kernel */
} nearmem_op_t;

/*
 * ============================================================
 * Initialization and Cleanup
 * ============================================================
 */

/*
 * nearmem_init - Initialize near-memory computing context
 * @ctx:         Context to initialize
 * @device_path: Path to pseudoscopic block device (e.g., "/dev/psdisk0")
 * @cuda_device: CUDA device index (usually 0, or matching GPU)
 *
 * Opens the pseudoscopic device, mmaps VRAM, and establishes
 * CUDA context for GPU-side operations.
 *
 * Returns: NEARMEM_OK on success, error code on failure
 */
nearmem_error_t nearmem_init(nearmem_ctx_t *ctx, 
                              const char *device_path,
                              int cuda_device);

/*
 * nearmem_init_auto - Auto-detect pseudoscopic device
 * @ctx:         Context to initialize
 *
 * Scans for /dev/psdisk* and matching CUDA device.
 *
 * Returns: NEARMEM_OK on success, error code on failure
 */
nearmem_error_t nearmem_init_auto(nearmem_ctx_t *ctx);

/*
 * nearmem_shutdown - Clean up context
 * @ctx: Context to destroy
 */
void nearmem_shutdown(nearmem_ctx_t *ctx);

/*
 * nearmem_get_capacity - Get total VRAM capacity
 * @ctx: Initialized context
 *
 * Returns: Capacity in bytes, or 0 on error
 */
size_t nearmem_get_capacity(nearmem_ctx_t *ctx);

/*
 * ============================================================
 * Memory Region Management
 * ============================================================
 */

/*
 * nearmem_alloc - Allocate a region of shared VRAM
 * @ctx:    Initialized context
 * @region: Output region handle
 * @size:   Size in bytes (will be rounded up to page size)
 *
 * The returned region has valid cpu_ptr and gpu_ptr pointing
 * to the same physical memory.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_alloc(nearmem_ctx_t *ctx,
                               nearmem_region_t *region,
                               size_t size);

/*
 * nearmem_free - Free a region
 * @ctx:    Context
 * @region: Region to free
 */
void nearmem_free(nearmem_ctx_t *ctx, nearmem_region_t *region);

/*
 * nearmem_map_offset - Map a specific VRAM offset
 * @ctx:    Context
 * @region: Output region handle
 * @offset: Byte offset within VRAM
 * @size:   Size to map
 *
 * For advanced use - maps existing VRAM region.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_map_offset(nearmem_ctx_t *ctx,
                                    nearmem_region_t *region,
                                    uint64_t offset,
                                    size_t size);

/*
 * ============================================================
 * Synchronization
 * ============================================================
 */

/*
 * nearmem_sync - Synchronize CPU and GPU views
 * @ctx:  Context
 * @type: Synchronization type
 *
 * CPU_TO_GPU: Ensures CPU writes are visible to GPU
 *   - Flushes CPU write-combine buffers
 *   - Issues memory fence
 *
 * GPU_TO_CPU: Ensures GPU writes are visible to CPU
 *   - Waits for GPU operations to complete
 *   - Invalidates CPU caches for region
 *
 * FULL: Both directions
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_sync(nearmem_ctx_t *ctx, nearmem_sync_t type);

/*
 * nearmem_sync_region - Synchronize specific region
 * @ctx:    Context
 * @region: Region to synchronize
 * @type:   Synchronization type
 *
 * More efficient than full sync when operating on subset.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_sync_region(nearmem_ctx_t *ctx,
                                     nearmem_region_t *region,
                                     nearmem_sync_t type);

/*
 * ============================================================
 * Built-in Operations (GPU-accelerated, in-place)
 * ============================================================
 */

/*
 * nearmem_memset - Fill region with value (GPU-accelerated)
 * @ctx:    Context
 * @region: Target region
 * @value:  Byte value to fill
 * @offset: Offset within region
 * @size:   Bytes to fill (0 = entire region)
 *
 * ~300 GB/s on modern GPUs vs ~12 GB/s over PCIe.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_memset(nearmem_ctx_t *ctx,
                                nearmem_region_t *region,
                                uint8_t value,
                                size_t offset,
                                size_t size);

/*
 * nearmem_memcpy - Copy within VRAM (no PCIe traffic)
 * @ctx:        Context
 * @dst_region: Destination region
 * @dst_offset: Offset in destination
 * @src_region: Source region
 * @src_offset: Offset in source
 * @size:       Bytes to copy
 *
 * Internal GPU bandwidth, not limited by PCIe.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_memcpy(nearmem_ctx_t *ctx,
                                nearmem_region_t *dst_region,
                                size_t dst_offset,
                                nearmem_region_t *src_region,
                                size_t src_offset,
                                size_t size);

/*
 * nearmem_find - Find pattern in region
 * @ctx:        Context
 * @region:     Region to search
 * @pattern:    Pattern bytes (can be in system RAM)
 * @pattern_len: Pattern length
 * @result:     Output - offset of first match, or -1
 *
 * GPU-parallel search, much faster than CPU for large regions.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_find(nearmem_ctx_t *ctx,
                              nearmem_region_t *region,
                              const void *pattern,
                              size_t pattern_len,
                              int64_t *result);

/*
 * nearmem_count_matches - Count pattern occurrences
 * @ctx:        Context
 * @region:     Region to search
 * @pattern:    Pattern bytes
 * @pattern_len: Pattern length
 * @count:      Output - number of matches
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_count_matches(nearmem_ctx_t *ctx,
                                       nearmem_region_t *region,
                                       const void *pattern,
                                       size_t pattern_len,
                                       uint64_t *count);

/*
 * nearmem_transform - Apply byte transformation
 * @ctx:    Context
 * @region: Region to transform (in-place)
 * @lut:    256-byte lookup table
 *
 * Each byte b becomes lut[b]. Useful for case conversion,
 * encoding transforms, etc.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_transform(nearmem_ctx_t *ctx,
                                   nearmem_region_t *region,
                                   const uint8_t lut[256]);

/*
 * nearmem_histogram - Compute byte histogram
 * @ctx:       Context
 * @region:    Region to analyze
 * @histogram: Output - 256 uint64_t counts
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_histogram(nearmem_ctx_t *ctx,
                                   nearmem_region_t *region,
                                   uint64_t histogram[256]);

/*
 * nearmem_sort_u32 - Sort 32-bit integers in-place
 * @ctx:    Context
 * @region: Region containing uint32_t array
 * @count:  Number of elements
 *
 * GPU radix sort - ~10 GB/s throughput.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_sort_u32(nearmem_ctx_t *ctx,
                                  nearmem_region_t *region,
                                  size_t count);

/*
 * nearmem_reduce_sum_f32 - Sum float array
 * @ctx:    Context
 * @region: Region containing float array
 * @count:  Number of elements
 * @result: Output sum
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_reduce_sum_f32(nearmem_ctx_t *ctx,
                                        nearmem_region_t *region,
                                        size_t count,
                                        float *result);

/*
 * ============================================================
 * Custom Kernel Execution
 * ============================================================
 */

/*
 * Kernel function signature for custom operations.
 *
 * The kernel receives:
 *   - data: Pointer to start of region (GPU address space)
 *   - size: Total size in bytes
 *   - arg:  User argument (copied to GPU constant memory)
 */
typedef void (*nearmem_kernel_fn)(void *data, size_t size, void *arg);

/*
 * nearmem_launch_custom - Launch user-defined kernel
 * @ctx:         Context
 * @region:      Region to operate on
 * @kernel:      CUDA kernel function
 * @arg:         Argument to pass (will be copied)
 * @arg_size:    Size of argument
 * @block_size:  CUDA block size (0 = auto)
 *
 * For operations not covered by built-ins.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_launch_custom(nearmem_ctx_t *ctx,
                                       nearmem_region_t *region,
                                       nearmem_kernel_fn kernel,
                                       void *arg,
                                       size_t arg_size,
                                       int block_size);

/*
 * ============================================================
 * Streaming Interface
 * ============================================================
 *
 * For processing data larger than VRAM, we support streaming:
 * CPU writes chunk → GPU processes → CPU reads results → repeat
 */

typedef struct nearmem_stream {
    nearmem_ctx_t       *ctx;
    nearmem_region_t    buffers[2];     /* Double-buffer */
    int                 active_buffer;
    size_t              chunk_size;
    void                *cuda_stream;   /* CUstream (opaque) */
} nearmem_stream_t;

/*
 * nearmem_stream_create - Create streaming context
 * @ctx:        Near-memory context
 * @stream:     Output stream handle
 * @chunk_size: Size of each buffer
 *
 * Double-buffering allows overlap of PCIe transfer and GPU compute.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_stream_create(nearmem_ctx_t *ctx,
                                       nearmem_stream_t *stream,
                                       size_t chunk_size);

/*
 * nearmem_stream_destroy - Destroy streaming context
 */
void nearmem_stream_destroy(nearmem_stream_t *stream);

/*
 * nearmem_stream_get_buffer - Get current write buffer
 * @stream: Stream context
 *
 * Returns: CPU pointer to write into
 */
void *nearmem_stream_get_buffer(nearmem_stream_t *stream);

/*
 * nearmem_stream_submit - Submit buffer for GPU processing
 * @stream: Stream context
 * @size:   Bytes written to buffer
 * @op:     Operation to perform
 * @arg:    Operation argument
 *
 * Swaps buffers and launches GPU kernel asynchronously.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_stream_submit(nearmem_stream_t *stream,
                                       size_t size,
                                       nearmem_op_t op,
                                       void *arg);

/*
 * nearmem_stream_wait - Wait for results
 * @stream: Stream context
 *
 * Blocks until GPU processing of previous buffer completes.
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nearmem_stream_wait(nearmem_stream_t *stream);

/*
 * ============================================================
 * Utility Functions
 * ============================================================
 */

/*
 * nearmem_strerror - Get error description
 * @err: Error code
 *
 * Returns: Human-readable error string
 */
const char *nearmem_strerror(nearmem_error_t err);

/*
 * nearmem_get_stats - Get performance statistics
 * @ctx:              Context
 * @bytes_to_gpu:     Output - bytes written by CPU
 * @bytes_from_gpu:   Output - bytes read by CPU
 * @gpu_operations:   Output - kernel launches
 * @gpu_time_us:      Output - total GPU time in microseconds
 */
void nearmem_get_stats(nearmem_ctx_t *ctx,
                       uint64_t *bytes_to_gpu,
                       uint64_t *bytes_from_gpu,
                       uint64_t *gpu_operations,
                       uint64_t *gpu_time_us);

#ifdef __cplusplus
}
#endif

#endif /* _NEARMEM_H_ */
