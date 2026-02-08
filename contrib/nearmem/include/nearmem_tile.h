/*
 * nearmem_tile.h - Tiled Near-Memory Computing API
 *
 * Extending NVIDIA's CUDA Tile paradigm across the CPU/GPU boundary.
 *
 * The Insight:
 * ------------
 * CUDA Tile abstracts GPU memory hierarchy (global → shared → registers).
 * We extend this to span the heterogeneous memory landscape:
 *
 *   System RAM → VRAM (BAR1) → GPU Shared → Registers
 *      ↑            ↑              ↑           ↑
 *     CPU        near-mem      tile cache   compute
 *    writes      staging        (fast)     (fastest)
 *
 * The programmer thinks in tiles. The runtime handles:
 *   - Async prefetching from CPU to VRAM staging
 *   - Cooperative loading from VRAM to shared memory
 *   - Double-buffering for compute/transfer overlap
 *   - Coherency barriers at tile boundaries
 *
 * Usage Pattern:
 * -------------
 *   // Define a 2D tile over VRAM data
 *   nm_tile_2d_t tile;
 *   nm_tile_2d_init(&tile, region, width, height, tile_w, tile_h, dtype);
 *
 *   // Iterate over tiles (runtime handles prefetch)
 *   nm_tile_iterator_t iter;
 *   nm_tile_for_each(&iter, &tile) {
 *       // Get tile data in GPU shared memory (already prefetched!)
 *       float* data = nm_tile_get_shared(&iter);
 *       
 *       // Your algorithm here - data is local and fast
 *       process(data, iter.tile_w, iter.tile_h);
 *       
 *       // Commit changes (runtime handles writeback)
 *       nm_tile_commit(&iter);
 *   }
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#ifndef _NEARMEM_TILE_H_
#define _NEARMEM_TILE_H_

#include "nearmem.h"
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ============================================================
 * Data Types
 * ============================================================
 */

/* Supported element types for typed tile access */
typedef enum {
    NM_DTYPE_U8,
    NM_DTYPE_I8,
    NM_DTYPE_U16,
    NM_DTYPE_I16,
    NM_DTYPE_U32,
    NM_DTYPE_I32,
    NM_DTYPE_U64,
    NM_DTYPE_I64,
    NM_DTYPE_F16,
    NM_DTYPE_BF16,
    NM_DTYPE_F32,
    NM_DTYPE_F64,
    NM_DTYPE_RAW,   /* Untyped bytes */
} nm_dtype_t;

/* Memory tier hints */
typedef enum {
    NM_TIER_AUTO,       /* Runtime decides */
    NM_TIER_CPU,        /* Force system RAM */
    NM_TIER_VRAM,       /* Force VRAM staging */
    NM_TIER_SHARED,     /* Force GPU shared memory */
} nm_tier_hint_t;

/* Access patterns for prefetch optimization */
typedef enum {
    NM_ACCESS_SEQUENTIAL,   /* Row-major sequential */
    NM_ACCESS_STRIDED,      /* Column-major or strided */
    NM_ACCESS_RANDOM,       /* Unpredictable */
    NM_ACCESS_TEMPORAL,     /* Will be reused soon */
    NM_ACCESS_STREAMING,    /* Use once, don't cache */
} nm_access_pattern_t;

/* Tile operation modes */
typedef enum {
    NM_TILE_READ,       /* Read-only access */
    NM_TILE_WRITE,      /* Write-only (no prefetch needed) */
    NM_TILE_READWRITE,  /* Read-modify-write */
    NM_TILE_REDUCE,     /* Reduction (special handling) */
} nm_tile_mode_t;

/*
 * ============================================================
 * Tile Descriptor Structures
 * ============================================================
 */

/*
 * 1D Tile Descriptor
 * For vectors, streams, sequences
 */
typedef struct nm_tile_1d {
    /* Underlying region */
    nearmem_region_t    *region;
    nearmem_ctx_t       *ctx;
    
    /* Geometry */
    size_t              length;         /* Total elements */
    size_t              tile_size;      /* Elements per tile */
    size_t              element_size;   /* Bytes per element */
    nm_dtype_t          dtype;
    
    /* Tiling state */
    size_t              num_tiles;
    size_t              stride;         /* Bytes between tiles (usually tile_size * element_size) */
    
    /* Runtime hints */
    nm_access_pattern_t access;
    nm_tile_mode_t      mode;
    nm_tier_hint_t      tier_hint;
    
    /* Prefetch state */
    int                 prefetch_depth; /* Tiles to prefetch ahead */
    void                *prefetch_buffer[2];  /* Double buffer */
    int                 active_buffer;
    
    /* CUDA resources */
    void                *stream;        /* CUstream for async ops */
    void                *event;         /* Completion event */
} nm_tile_1d_t;

/*
 * 2D Tile Descriptor
 * For matrices, images, feature maps
 */
typedef struct nm_tile_2d {
    /* Underlying region */
    nearmem_region_t    *region;
    nearmem_ctx_t       *ctx;
    
    /* Geometry */
    size_t              width;          /* Total columns */
    size_t              height;         /* Total rows */
    size_t              tile_w;         /* Tile width */
    size_t              tile_h;         /* Tile height */
    size_t              element_size;
    nm_dtype_t          dtype;
    
    /* Layout */
    size_t              pitch;          /* Row stride in bytes */
    size_t              num_tiles_x;
    size_t              num_tiles_y;
    size_t              num_tiles;
    
    /* Runtime hints */
    nm_access_pattern_t access;
    nm_tile_mode_t      mode;
    nm_tier_hint_t      tier_hint;
    
    /* Halo/overlap for stencil operations */
    int                 halo_x;         /* Overlap in x */
    int                 halo_y;         /* Overlap in y */
    
    /* Prefetch state */
    int                 prefetch_depth;
    void                *prefetch_buffer[2];
    size_t              buffer_size;
    int                 active_buffer;
    
    /* CUDA resources */
    void                *stream;
    void                *event;
} nm_tile_2d_t;

/*
 * 3D Tile Descriptor  
 * For volumes, video, 3D convolutions
 */
typedef struct nm_tile_3d {
    nearmem_region_t    *region;
    nearmem_ctx_t       *ctx;
    
    size_t              width, height, depth;
    size_t              tile_w, tile_h, tile_d;
    size_t              element_size;
    nm_dtype_t          dtype;
    
    size_t              pitch;          /* Row stride */
    size_t              slice_pitch;    /* Slice stride */
    size_t              num_tiles_x, num_tiles_y, num_tiles_z;
    size_t              num_tiles;
    
    nm_access_pattern_t access;
    nm_tile_mode_t      mode;
    nm_tier_hint_t      tier_hint;
    
    int                 halo_x, halo_y, halo_z;
    
    int                 prefetch_depth;
    void                *prefetch_buffer[2];
    size_t              buffer_size;
    int                 active_buffer;
    
    void                *stream;
    void                *event;
} nm_tile_3d_t;

/*
 * ============================================================
 * Tile Iterator
 * ============================================================
 * Used during tile traversal to track position and provide access.
 */

typedef struct nm_tile_iterator {
    /* Parent tile descriptor */
    union {
        nm_tile_1d_t    *tile_1d;
        nm_tile_2d_t    *tile_2d;
        nm_tile_3d_t    *tile_3d;
        void            *tile_ptr;
    };
    int                 ndim;           /* 1, 2, or 3 */
    
    /* Current position */
    size_t              tile_idx;       /* Linear tile index */
    size_t              tile_x, tile_y, tile_z;  /* Tile coordinates */
    
    /* Current tile bounds (in elements) */
    size_t              start_x, start_y, start_z;
    size_t              end_x, end_y, end_z;
    size_t              current_tile_w, current_tile_h, current_tile_d;
    
    /* Data pointers */
    void                *cpu_ptr;       /* CPU-accessible (via BAR1) */
    void                *gpu_ptr;       /* GPU device pointer */
    void                *shared_ptr;    /* GPU shared memory (if loaded) */
    
    /* State flags */
    bool                prefetched;     /* Data in prefetch buffer */
    bool                in_shared;      /* Data loaded to shared memory */
    bool                dirty;          /* Modified, needs writeback */
    bool                last_tile;      /* Final tile in iteration */
    
    /* Timing (optional) */
    double              prefetch_time_us;
    double              compute_time_us;
    double              writeback_time_us;
} nm_tile_iterator_t;

/*
 * ============================================================
 * Initialization Functions
 * ============================================================
 */

/*
 * nm_tile_1d_init - Initialize 1D tile descriptor
 * @tile:       Tile descriptor to initialize
 * @ctx:        Near-memory context
 * @region:     VRAM region containing data
 * @length:     Total number of elements
 * @tile_size:  Elements per tile
 * @dtype:      Element data type
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nm_tile_1d_init(nm_tile_1d_t *tile,
                                 nearmem_ctx_t *ctx,
                                 nearmem_region_t *region,
                                 size_t length,
                                 size_t tile_size,
                                 nm_dtype_t dtype);

/*
 * nm_tile_2d_init - Initialize 2D tile descriptor
 * @tile:       Tile descriptor to initialize
 * @ctx:        Near-memory context
 * @region:     VRAM region containing data
 * @width:      Total columns
 * @height:     Total rows
 * @pitch:      Row stride in bytes (0 = tightly packed)
 * @tile_w:     Tile width
 * @tile_h:     Tile height
 * @dtype:      Element data type
 *
 * Returns: NEARMEM_OK on success
 */
nearmem_error_t nm_tile_2d_init(nm_tile_2d_t *tile,
                                 nearmem_ctx_t *ctx,
                                 nearmem_region_t *region,
                                 size_t width,
                                 size_t height,
                                 size_t pitch,
                                 size_t tile_w,
                                 size_t tile_h,
                                 nm_dtype_t dtype);

/*
 * nm_tile_3d_init - Initialize 3D tile descriptor
 */
nearmem_error_t nm_tile_3d_init(nm_tile_3d_t *tile,
                                 nearmem_ctx_t *ctx,
                                 nearmem_region_t *region,
                                 size_t width,
                                 size_t height,
                                 size_t depth,
                                 size_t pitch,
                                 size_t slice_pitch,
                                 size_t tile_w,
                                 size_t tile_h,
                                 size_t tile_d,
                                 nm_dtype_t dtype);

/*
 * nm_tile_destroy - Clean up tile descriptor
 */
void nm_tile_1d_destroy(nm_tile_1d_t *tile);
void nm_tile_2d_destroy(nm_tile_2d_t *tile);
void nm_tile_3d_destroy(nm_tile_3d_t *tile);

/*
 * ============================================================
 * Configuration Functions
 * ============================================================
 */

/*
 * nm_tile_set_access - Set access pattern hint
 * Helps runtime optimize prefetching strategy.
 */
void nm_tile_1d_set_access(nm_tile_1d_t *tile, nm_access_pattern_t access);
void nm_tile_2d_set_access(nm_tile_2d_t *tile, nm_access_pattern_t access);
void nm_tile_3d_set_access(nm_tile_3d_t *tile, nm_access_pattern_t access);

/*
 * nm_tile_set_mode - Set read/write mode
 * Enables optimizations (skip prefetch for write-only, etc.)
 */
void nm_tile_1d_set_mode(nm_tile_1d_t *tile, nm_tile_mode_t mode);
void nm_tile_2d_set_mode(nm_tile_2d_t *tile, nm_tile_mode_t mode);
void nm_tile_3d_set_mode(nm_tile_3d_t *tile, nm_tile_mode_t mode);

/*
 * nm_tile_set_halo - Set halo/overlap for stencil operations
 * When accessing tile[i], you get tile[i-halo] to tile[i+tile_size+halo]
 */
void nm_tile_2d_set_halo(nm_tile_2d_t *tile, int halo_x, int halo_y);
void nm_tile_3d_set_halo(nm_tile_3d_t *tile, int halo_x, int halo_y, int halo_z);

/*
 * nm_tile_set_prefetch_depth - Set how many tiles to prefetch ahead
 * Default is 2 (double buffering). 0 disables prefetching.
 */
void nm_tile_1d_set_prefetch(nm_tile_1d_t *tile, int depth);
void nm_tile_2d_set_prefetch(nm_tile_2d_t *tile, int depth);
void nm_tile_3d_set_prefetch(nm_tile_3d_t *tile, int depth);

/*
 * ============================================================
 * Iteration Functions
 * ============================================================
 */

/*
 * nm_tile_begin - Start tile iteration
 * @iter:  Iterator to initialize
 * @tile:  Tile descriptor
 *
 * Prefetches first tile(s) and prepares for traversal.
 */
nearmem_error_t nm_tile_1d_begin(nm_tile_iterator_t *iter, nm_tile_1d_t *tile);
nearmem_error_t nm_tile_2d_begin(nm_tile_iterator_t *iter, nm_tile_2d_t *tile);
nearmem_error_t nm_tile_3d_begin(nm_tile_iterator_t *iter, nm_tile_3d_t *tile);

/*
 * nm_tile_next - Advance to next tile
 * @iter:  Iterator
 *
 * Handles:
 *   - Writeback of current tile (if dirty)
 *   - Prefetch of upcoming tile(s)
 *   - Wait for prefetch completion
 *   - Update iterator position
 *
 * Returns: true if more tiles remain, false when done
 */
bool nm_tile_next(nm_tile_iterator_t *iter);

/*
 * nm_tile_done - Check if iteration complete
 */
static inline bool nm_tile_done(nm_tile_iterator_t *iter) {
    return iter->last_tile && iter->tile_idx > 0;
}

/*
 * nm_tile_end - Finish iteration, flush any pending writes
 */
void nm_tile_end(nm_tile_iterator_t *iter);

/*
 * ============================================================
 * Data Access Functions
 * ============================================================
 */

/*
 * nm_tile_get_cpu - Get CPU pointer to current tile
 * 
 * Returns pointer via BAR1 mapping. Use for CPU-side operations.
 * Note: May have higher latency than GPU access.
 */
void *nm_tile_get_cpu(nm_tile_iterator_t *iter);

/*
 * nm_tile_get_gpu - Get GPU device pointer to current tile
 *
 * Returns CUDA device pointer for kernel launch.
 * Data is in VRAM, ready for GPU processing.
 */
void *nm_tile_get_gpu(nm_tile_iterator_t *iter);

/*
 * nm_tile_get_shared - Get pointer in GPU shared memory
 *
 * For maximum performance, tile data can be cooperatively loaded
 * to shared memory. This returns a pointer valid within the kernel.
 *
 * Note: Call nm_tile_load_shared() first from your kernel.
 */
void *nm_tile_get_shared(nm_tile_iterator_t *iter);

/*
 * nm_tile_mark_dirty - Mark tile as modified
 *
 * Must be called if you modify tile data and want changes persisted.
 * Writeback happens on nm_tile_next() or nm_tile_end().
 */
void nm_tile_mark_dirty(nm_tile_iterator_t *iter);

/*
 * nm_tile_prefetch - Manually trigger prefetch for tile
 * @iter:       Iterator
 * @tile_idx:   Tile index to prefetch
 *
 * Useful for non-sequential access patterns.
 */
void nm_tile_prefetch(nm_tile_iterator_t *iter, size_t tile_idx);

/*
 * ============================================================
 * Synchronization Functions
 * ============================================================
 */

/*
 * nm_tile_sync_cpu - Ensure CPU writes visible to GPU
 *
 * Call after CPU modifications before GPU kernel launch.
 */
void nm_tile_sync_cpu(nm_tile_iterator_t *iter);

/*
 * nm_tile_sync_gpu - Ensure GPU writes visible to CPU
 *
 * Call after GPU kernel, before CPU reads results.
 */
void nm_tile_sync_gpu(nm_tile_iterator_t *iter);

/*
 * nm_tile_barrier - Full synchronization barrier
 *
 * Waits for all pending operations (prefetch, writeback, compute).
 */
void nm_tile_barrier(nm_tile_iterator_t *iter);

/*
 * ============================================================
 * Convenience Macros
 * ============================================================
 */

/* 
 * Iterate over all tiles with automatic prefetch and writeback
 *
 * Usage:
 *   nm_tile_for_each(iter, &my_tile) {
 *       float *data = nm_tile_get_gpu(&iter);
 *       my_kernel<<<...>>>(data, iter.current_tile_w, iter.current_tile_h);
 *       nm_tile_mark_dirty(&iter);
 *   }
 */
#define nm_tile_for_each_1d(iter_ptr, tile_ptr) \
    for (nm_tile_1d_begin(iter_ptr, tile_ptr); \
         !nm_tile_done(iter_ptr); \
         nm_tile_next(iter_ptr))

#define nm_tile_for_each_2d(iter_ptr, tile_ptr) \
    for (nm_tile_2d_begin(iter_ptr, tile_ptr); \
         !nm_tile_done(iter_ptr); \
         nm_tile_next(iter_ptr))

#define nm_tile_for_each_3d(iter_ptr, tile_ptr) \
    for (nm_tile_3d_begin(iter_ptr, tile_ptr); \
         !nm_tile_done(iter_ptr); \
         nm_tile_next(iter_ptr))

/* Generic dispatch based on tile dimension */
#define nm_tile_for_each(iter_ptr, tile_ptr) \
    _Generic((tile_ptr), \
        nm_tile_1d_t*: nm_tile_for_each_1d, \
        nm_tile_2d_t*: nm_tile_for_each_2d, \
        nm_tile_3d_t*: nm_tile_for_each_3d \
    )(iter_ptr, tile_ptr)

/*
 * ============================================================
 * Utility Functions
 * ============================================================
 */

/*
 * nm_dtype_size - Get size in bytes for data type
 */
size_t nm_dtype_size(nm_dtype_t dtype);

/*
 * nm_dtype_name - Get string name for data type
 */
const char *nm_dtype_name(nm_dtype_t dtype);

/*
 * nm_tile_stats - Get iteration statistics
 */
typedef struct {
    size_t          tiles_processed;
    size_t          bytes_prefetched;
    size_t          bytes_written_back;
    double          total_prefetch_us;
    double          total_compute_us;
    double          total_writeback_us;
    double          prefetch_bandwidth_gbps;
    double          writeback_bandwidth_gbps;
} nm_tile_stats_t;

void nm_tile_get_stats(nm_tile_iterator_t *iter, nm_tile_stats_t *stats);

/*
 * nm_tile_print_stats - Print statistics summary
 */
void nm_tile_print_stats(nm_tile_stats_t *stats);

/*
 * ============================================================
 * CUDA Kernel Helpers (Device Code)
 * ============================================================
 * These are meant to be called from within CUDA kernels.
 */

#ifdef __CUDACC__

/*
 * nm_tile_cooperative_load - Load tile to shared memory cooperatively
 *
 * All threads in block participate in loading.
 * Call this at start of kernel, before accessing tile data.
 *
 * @shared_ptr:  Pointer to shared memory buffer
 * @global_ptr:  Pointer to global memory (from nm_tile_get_gpu)
 * @tile_size:   Size in bytes
 */
__device__ void nm_tile_cooperative_load(void *shared_ptr,
                                          const void *global_ptr,
                                          size_t tile_size);

/*
 * nm_tile_cooperative_store - Store tile from shared memory cooperatively
 *
 * All threads participate in storing.
 * Call before kernel exit if tile was modified.
 */
__device__ void nm_tile_cooperative_store(void *global_ptr,
                                           const void *shared_ptr,
                                           size_t tile_size);

/*
 * nm_tile_load_2d - Load 2D tile with proper striding
 */
__device__ void nm_tile_load_2d(void *shared_ptr,
                                 const void *global_ptr,
                                 size_t tile_w,
                                 size_t tile_h,
                                 size_t global_pitch,
                                 size_t element_size);

/*
 * nm_tile_store_2d - Store 2D tile with proper striding
 */
__device__ void nm_tile_store_2d(void *global_ptr,
                                  const void *shared_ptr,
                                  size_t tile_w,
                                  size_t tile_h,
                                  size_t global_pitch,
                                  size_t element_size);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* _NEARMEM_TILE_H_ */
