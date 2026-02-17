/*
 * nearmem_tile.c - Tiled Near-Memory Computing Implementation
 *
 * The Core Machinery:
 * -------------------
 * 1. Prefetch: Async load from VRAM staging to prefetch buffer
 * 2. Process: User code operates on current buffer
 * 3. Writeback: Async store modified data back to VRAM
 * 4. Swap: Exchange prefetch buffers (double-buffering)
 *
 * The magic: while GPU computes on buffer A, we're prefetching
 * into buffer B. When compute finishes, we swap and repeat.
 * This hides memory latency behind compute.
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "nearmem_tile.h"

/* CUDA driver API (dynamically loaded via nearmem.c) */
typedef void* CUstream;
typedef void* CUevent;
typedef int CUresult;

extern CUresult (*cuStreamCreate)(CUstream*, unsigned int);
extern CUresult (*cuStreamDestroy)(CUstream);
extern CUresult (*cuStreamSynchronize)(CUstream);
extern CUresult (*cuEventCreate)(CUevent*, unsigned int);
extern CUresult (*cuEventDestroy)(CUevent);
extern CUresult (*cuEventRecord)(CUevent, CUstream);
extern CUresult (*cuEventSynchronize)(CUevent);
extern CUresult (*cuMemcpyAsync)(void*, const void*, size_t, CUstream);
extern CUresult (*cuMemcpyDtoDAsync)(void*, void*, size_t, CUstream);

#define CUDA_SUCCESS 0

/*
 * Timing helper
 */
static double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

/*
 * Data type sizes
 */
static const size_t dtype_sizes[] = {
    [NM_DTYPE_U8]   = 1,
    [NM_DTYPE_I8]   = 1,
    [NM_DTYPE_U16]  = 2,
    [NM_DTYPE_I16]  = 2,
    [NM_DTYPE_U32]  = 4,
    [NM_DTYPE_I32]  = 4,
    [NM_DTYPE_U64]  = 8,
    [NM_DTYPE_I64]  = 8,
    [NM_DTYPE_F16]  = 2,
    [NM_DTYPE_BF16] = 2,
    [NM_DTYPE_F32]  = 4,
    [NM_DTYPE_F64]  = 8,
    [NM_DTYPE_RAW]  = 1,
};

static const char *dtype_names[] = {
    [NM_DTYPE_U8]   = "uint8",
    [NM_DTYPE_I8]   = "int8",
    [NM_DTYPE_U16]  = "uint16",
    [NM_DTYPE_I16]  = "int16",
    [NM_DTYPE_U32]  = "uint32",
    [NM_DTYPE_I32]  = "int32",
    [NM_DTYPE_U64]  = "uint64",
    [NM_DTYPE_I64]  = "int64",
    [NM_DTYPE_F16]  = "float16",
    [NM_DTYPE_BF16] = "bfloat16",
    [NM_DTYPE_F32]  = "float32",
    [NM_DTYPE_F64]  = "float64",
    [NM_DTYPE_RAW]  = "raw",
};

size_t nm_dtype_size(nm_dtype_t dtype)
{
    if (dtype >= 0 && dtype <= NM_DTYPE_RAW)
        return dtype_sizes[dtype];
    return 1;
}

const char *nm_dtype_name(nm_dtype_t dtype)
{
    if (dtype >= 0 && dtype <= NM_DTYPE_RAW)
        return dtype_names[dtype];
    return "unknown";
}

/*
 * Ceiling division helper
 */
static inline size_t div_ceil(size_t a, size_t b)
{
    return (a + b - 1) / b;
}

/*
 * ============================================================
 * 1D Tile Implementation
 * ============================================================
 */

nearmem_error_t nm_tile_1d_init(nm_tile_1d_t *tile,
                                 nearmem_ctx_t *ctx,
                                 nearmem_region_t *region,
                                 size_t length,
                                 size_t tile_size,
                                 nm_dtype_t dtype)
{
    if (!tile || !ctx || !region || length == 0 || tile_size == 0)
        return NEARMEM_ERROR_INVALID;
    
    memset(tile, 0, sizeof(*tile));
    
    tile->ctx = ctx;
    tile->region = region;
    tile->length = length;
    tile->tile_size = tile_size;
    tile->element_size = nm_dtype_size(dtype);
    tile->dtype = dtype;
    
    tile->num_tiles = div_ceil(length, tile_size);
    tile->stride = tile_size * tile->element_size;
    
    /* Defaults */
    tile->access = NM_ACCESS_SEQUENTIAL;
    tile->mode = NM_TILE_READWRITE;
    tile->tier_hint = NM_TIER_AUTO;
    tile->prefetch_depth = 2;  /* Double buffering */
    
    /* Allocate prefetch buffers (in system RAM for now) */
    size_t buffer_size = tile->stride;
    tile->prefetch_buffer[0] = malloc(buffer_size);
    tile->prefetch_buffer[1] = malloc(buffer_size);
    
    if (!tile->prefetch_buffer[0] || !tile->prefetch_buffer[1]) {
        free(tile->prefetch_buffer[0]);
        free(tile->prefetch_buffer[1]);
        return NEARMEM_ERROR_INIT;
    }
    
    /* Create CUDA stream for async operations */
    if (cuStreamCreate) {
        cuStreamCreate((CUstream*)&tile->stream, 0);
        cuEventCreate((CUevent*)&tile->event, 0);
    }
    
    return NEARMEM_OK;
}

void nm_tile_1d_destroy(nm_tile_1d_t *tile)
{
    if (!tile)
        return;
    
    if (tile->stream && cuStreamDestroy)
        cuStreamDestroy(tile->stream);
    if (tile->event && cuEventDestroy)
        cuEventDestroy(tile->event);
    
    free(tile->prefetch_buffer[0]);
    free(tile->prefetch_buffer[1]);
    
    memset(tile, 0, sizeof(*tile));
}

void nm_tile_1d_set_access(nm_tile_1d_t *tile, nm_access_pattern_t access)
{
    if (tile) tile->access = access;
}

void nm_tile_1d_set_mode(nm_tile_1d_t *tile, nm_tile_mode_t mode)
{
    if (tile) tile->mode = mode;
}

void nm_tile_1d_set_prefetch(nm_tile_1d_t *tile, int depth)
{
    if (tile) tile->prefetch_depth = (depth > 0) ? depth : 0;
}

/*
 * ============================================================
 * 2D Tile Implementation
 * ============================================================
 */

nearmem_error_t nm_tile_2d_init(nm_tile_2d_t *tile,
                                 nearmem_ctx_t *ctx,
                                 nearmem_region_t *region,
                                 size_t width,
                                 size_t height,
                                 size_t pitch,
                                 size_t tile_w,
                                 size_t tile_h,
                                 nm_dtype_t dtype)
{
    if (!tile || !ctx || !region || 
        width == 0 || height == 0 || 
        tile_w == 0 || tile_h == 0)
        return NEARMEM_ERROR_INVALID;
    
    memset(tile, 0, sizeof(*tile));
    
    tile->ctx = ctx;
    tile->region = region;
    tile->width = width;
    tile->height = height;
    tile->tile_w = tile_w;
    tile->tile_h = tile_h;
    tile->element_size = nm_dtype_size(dtype);
    tile->dtype = dtype;
    
    /* Calculate pitch (row stride) */
    tile->pitch = (pitch > 0) ? pitch : width * tile->element_size;
    
    /* Calculate tile counts */
    tile->num_tiles_x = div_ceil(width, tile_w);
    tile->num_tiles_y = div_ceil(height, tile_h);
    tile->num_tiles = tile->num_tiles_x * tile->num_tiles_y;
    
    /* Defaults */
    tile->access = NM_ACCESS_SEQUENTIAL;
    tile->mode = NM_TILE_READWRITE;
    tile->tier_hint = NM_TIER_AUTO;
    tile->prefetch_depth = 2;
    tile->halo_x = 0;
    tile->halo_y = 0;
    
    /* Allocate prefetch buffers */
    tile->buffer_size = (tile_w + 2 * tile->halo_x) * 
                        (tile_h + 2 * tile->halo_y) * 
                        tile->element_size;
    
    tile->prefetch_buffer[0] = malloc(tile->buffer_size);
    tile->prefetch_buffer[1] = malloc(tile->buffer_size);
    
    if (!tile->prefetch_buffer[0] || !tile->prefetch_buffer[1]) {
        free(tile->prefetch_buffer[0]);
        free(tile->prefetch_buffer[1]);
        return NEARMEM_ERROR_INIT;
    }
    
    /* Create CUDA resources */
    if (cuStreamCreate) {
        cuStreamCreate((CUstream*)&tile->stream, 0);
        cuEventCreate((CUevent*)&tile->event, 0);
    }
    
    return NEARMEM_OK;
}

void nm_tile_2d_destroy(nm_tile_2d_t *tile)
{
    if (!tile)
        return;
    
    if (tile->stream && cuStreamDestroy)
        cuStreamDestroy(tile->stream);
    if (tile->event && cuEventDestroy)
        cuEventDestroy(tile->event);
    
    free(tile->prefetch_buffer[0]);
    free(tile->prefetch_buffer[1]);
    
    memset(tile, 0, sizeof(*tile));
}

void nm_tile_2d_set_access(nm_tile_2d_t *tile, nm_access_pattern_t access)
{
    if (tile) tile->access = access;
}

void nm_tile_2d_set_mode(nm_tile_2d_t *tile, nm_tile_mode_t mode)
{
    if (tile) tile->mode = mode;
}

void nm_tile_2d_set_halo(nm_tile_2d_t *tile, int halo_x, int halo_y)
{
    if (!tile)
        return;
    
    tile->halo_x = halo_x;
    tile->halo_y = halo_y;
    
    /* Reallocate buffers with halo */
    size_t new_size = (tile->tile_w + 2 * halo_x) * 
                      (tile->tile_h + 2 * halo_y) * 
                      tile->element_size;
    
    if (new_size != tile->buffer_size) {
        tile->buffer_size = new_size;
        tile->prefetch_buffer[0] = realloc(tile->prefetch_buffer[0], new_size);
        tile->prefetch_buffer[1] = realloc(tile->prefetch_buffer[1], new_size);
    }
}

void nm_tile_2d_set_prefetch(nm_tile_2d_t *tile, int depth)
{
    if (tile) tile->prefetch_depth = (depth > 0) ? depth : 0;
}

/*
 * ============================================================
 * 3D Tile Implementation (abbreviated)
 * ============================================================
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
                                 nm_dtype_t dtype)
{
    if (!tile || !ctx || !region)
        return NEARMEM_ERROR_INVALID;
    
    memset(tile, 0, sizeof(*tile));
    
    tile->ctx = ctx;
    tile->region = region;
    tile->width = width;
    tile->height = height;
    tile->depth = depth;
    tile->tile_w = tile_w;
    tile->tile_h = tile_h;
    tile->tile_d = tile_d;
    tile->element_size = nm_dtype_size(dtype);
    tile->dtype = dtype;
    
    tile->pitch = (pitch > 0) ? pitch : width * tile->element_size;
    tile->slice_pitch = (slice_pitch > 0) ? slice_pitch : tile->pitch * height;
    
    tile->num_tiles_x = div_ceil(width, tile_w);
    tile->num_tiles_y = div_ceil(height, tile_h);
    tile->num_tiles_z = div_ceil(depth, tile_d);
    tile->num_tiles = tile->num_tiles_x * tile->num_tiles_y * tile->num_tiles_z;
    
    tile->access = NM_ACCESS_SEQUENTIAL;
    tile->mode = NM_TILE_READWRITE;
    tile->tier_hint = NM_TIER_AUTO;
    tile->prefetch_depth = 2;
    
    tile->buffer_size = tile_w * tile_h * tile_d * tile->element_size;
    tile->prefetch_buffer[0] = malloc(tile->buffer_size);
    tile->prefetch_buffer[1] = malloc(tile->buffer_size);
    
    if (!tile->prefetch_buffer[0] || !tile->prefetch_buffer[1]) {
        free(tile->prefetch_buffer[0]);
        free(tile->prefetch_buffer[1]);
        return NEARMEM_ERROR_INIT;
    }
    
    if (cuStreamCreate) {
        cuStreamCreate((CUstream*)&tile->stream, 0);
        cuEventCreate((CUevent*)&tile->event, 0);
    }
    
    return NEARMEM_OK;
}

void nm_tile_3d_destroy(nm_tile_3d_t *tile)
{
    if (!tile)
        return;
    
    if (tile->stream && cuStreamDestroy)
        cuStreamDestroy(tile->stream);
    if (tile->event && cuEventDestroy)
        cuEventDestroy(tile->event);
    
    free(tile->prefetch_buffer[0]);
    free(tile->prefetch_buffer[1]);
    
    memset(tile, 0, sizeof(*tile));
}

void nm_tile_3d_set_access(nm_tile_3d_t *tile, nm_access_pattern_t access)
{
    if (tile) tile->access = access;
}

void nm_tile_3d_set_mode(nm_tile_3d_t *tile, nm_tile_mode_t mode)
{
    if (tile) tile->mode = mode;
}

void nm_tile_3d_set_halo(nm_tile_3d_t *tile, int halo_x, int halo_y, int halo_z)
{
    if (!tile)
        return;
    
    tile->halo_x = halo_x;
    tile->halo_y = halo_y;
    tile->halo_z = halo_z;
    
    size_t new_size = (tile->tile_w + 2 * halo_x) * 
                      (tile->tile_h + 2 * halo_y) * 
                      (tile->tile_d + 2 * halo_z) *
                      tile->element_size;
    
    if (new_size != tile->buffer_size) {
        tile->buffer_size = new_size;
        tile->prefetch_buffer[0] = realloc(tile->prefetch_buffer[0], new_size);
        tile->prefetch_buffer[1] = realloc(tile->prefetch_buffer[1], new_size);
    }
}

void nm_tile_3d_set_prefetch(nm_tile_3d_t *tile, int depth)
{
    if (tile) tile->prefetch_depth = (depth > 0) ? depth : 0;
}

/*
 * ============================================================
 * Iteration Implementation
 * ============================================================
 */

/*
 * Internal: Prefetch a tile into buffer
 */
static void tile_prefetch_2d(nm_tile_2d_t *tile, 
                              nm_tile_iterator_t *iter,
                              size_t tile_idx,
                              void *buffer)
{
    /* Calculate tile position */
    size_t tile_y = tile_idx / tile->num_tiles_x;
    size_t tile_x = tile_idx % tile->num_tiles_x;
    
    size_t start_x = tile_x * tile->tile_w;
    size_t start_y = tile_y * tile->tile_h;
    
    /* Handle halo (extend bounds, clamp to edges) */
    size_t fetch_start_x = (start_x > (size_t)tile->halo_x) ? 
                           start_x - tile->halo_x : 0;
    size_t fetch_start_y = (start_y > (size_t)tile->halo_y) ? 
                           start_y - tile->halo_y : 0;
    
    size_t fetch_end_x = start_x + tile->tile_w + tile->halo_x;
    size_t fetch_end_y = start_y + tile->tile_h + tile->halo_y;
    
    /* Boundary clamps: typically not taken for interior tiles */
    if (__builtin_expect(fetch_end_x > tile->width, 0))  fetch_end_x = tile->width;
    if (__builtin_expect(fetch_end_y > tile->height, 0)) fetch_end_y = tile->height;
    
    size_t fetch_w = fetch_end_x - fetch_start_x;
    size_t fetch_h = fetch_end_y - fetch_start_y;
    size_t row_bytes = fetch_w * tile->element_size;
    
    char *src_base = (char*)tile->region->cpu_ptr;
    char *dst_base = buffer;
    
    double start_time = get_time_us();
    
    /*
     * NOTE: We do NOT call nearmem_sync() per-tile.
     * The iteration begin function (nm_tile_2d_begin) handles the
     * initial sync. Per-tile syncs are redundant and expensive
     * for BAR1/WC-mapped memory where reads always hit VRAM.
     * A memory fence is sufficient to order reads after any prior
     * writes within the same iteration scope.
     */
    __sync_synchronize();
    
    for (size_t row = 0; row < fetch_h; row++) {
        size_t src_offset = (fetch_start_y + row) * tile->pitch + 
                            fetch_start_x * tile->element_size;
        size_t dst_offset = row * row_bytes;
        
        /* Prefetch next row's source data while copying current row */
        if (__builtin_expect(row + 1 < fetch_h, 1)) {
            size_t next_src = (fetch_start_y + row + 1) * tile->pitch + 
                              fetch_start_x * tile->element_size;
            __builtin_prefetch(src_base + next_src, 0, 1);
        }
        
        memcpy(dst_base + dst_offset, 
               src_base + src_offset, 
               row_bytes);
    }
    
    iter->prefetch_time_us += get_time_us() - start_time;
}

/*
 * Internal: Writeback a tile from buffer to VRAM
 */
static void tile_writeback_2d(nm_tile_2d_t *tile,
                               nm_tile_iterator_t *iter,
                               void *buffer)
{
    char *src_base = buffer;
    char *dst_base = (char*)tile->region->cpu_ptr;
    
    double start_time = get_time_us();
    
    /* Copy rows from buffer to VRAM */
    for (size_t row = 0; row < iter->current_tile_h; row++) {
        size_t dst_offset = (iter->start_y + row) * tile->pitch + 
                            iter->start_x * tile->element_size;
        size_t src_offset = row * iter->current_tile_w * tile->element_size;
        
        memcpy(dst_base + dst_offset,
               src_base + src_offset,
               iter->current_tile_w * tile->element_size);
    }
    
    /* Sync CPU → VRAM after writing */
    nearmem_sync(tile->ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    iter->writeback_time_us += get_time_us() - start_time;
}

/*
 * Begin iteration
 */
nearmem_error_t nm_tile_2d_begin(nm_tile_iterator_t *iter, nm_tile_2d_t *tile)
{
    if (!iter || !tile)
        return NEARMEM_ERROR_INVALID;
    
    memset(iter, 0, sizeof(*iter));
    
    iter->tile_2d = tile;
    iter->ndim = 2;
    iter->tile_idx = 0;
    iter->tile_x = 0;
    iter->tile_y = 0;
    
    /* Calculate bounds for first tile */
    iter->start_x = 0;
    iter->start_y = 0;
    iter->end_x = (tile->tile_w < tile->width) ? tile->tile_w : tile->width;
    iter->end_y = (tile->tile_h < tile->height) ? tile->tile_h : tile->height;
    iter->current_tile_w = iter->end_x - iter->start_x;
    iter->current_tile_h = iter->end_y - iter->start_y;
    
    /* Set pointers */
    iter->cpu_ptr = tile->region->cpu_ptr;
    iter->gpu_ptr = tile->region->gpu_ptr;
    
    /* Prefetch first tile if in read mode */
    if (tile->mode != NM_TILE_WRITE && tile->prefetch_depth > 0) {
        tile_prefetch_2d(tile, iter, 0, tile->prefetch_buffer[0]);
        iter->prefetched = true;
        tile->active_buffer = 0;
    }
    
    /* Prefetch second tile (double buffer) */
    if (tile->prefetch_depth > 1 && tile->num_tiles > 1) {
        tile_prefetch_2d(tile, iter, 1, tile->prefetch_buffer[1]);
    }
    
    iter->last_tile = (tile->num_tiles <= 1);
    
    return NEARMEM_OK;
}

nearmem_error_t nm_tile_1d_begin(nm_tile_iterator_t *iter, nm_tile_1d_t *tile)
{
    if (!iter || !tile)
        return NEARMEM_ERROR_INVALID;
    
    memset(iter, 0, sizeof(*iter));
    
    iter->tile_1d = tile;
    iter->ndim = 1;
    iter->tile_idx = 0;
    
    iter->start_x = 0;
    iter->end_x = (tile->tile_size < tile->length) ? tile->tile_size : tile->length;
    iter->current_tile_w = iter->end_x - iter->start_x;
    
    iter->cpu_ptr = tile->region->cpu_ptr;
    iter->gpu_ptr = tile->region->gpu_ptr;
    
    iter->last_tile = (tile->num_tiles <= 1);
    
    return NEARMEM_OK;
}

nearmem_error_t nm_tile_3d_begin(nm_tile_iterator_t *iter, nm_tile_3d_t *tile)
{
    if (!iter || !tile)
        return NEARMEM_ERROR_INVALID;
    
    memset(iter, 0, sizeof(*iter));
    
    iter->tile_3d = tile;
    iter->ndim = 3;
    iter->tile_idx = 0;
    iter->tile_x = iter->tile_y = iter->tile_z = 0;
    
    iter->start_x = iter->start_y = iter->start_z = 0;
    iter->end_x = (tile->tile_w < tile->width) ? tile->tile_w : tile->width;
    iter->end_y = (tile->tile_h < tile->height) ? tile->tile_h : tile->height;
    iter->end_z = (tile->tile_d < tile->depth) ? tile->tile_d : tile->depth;
    iter->current_tile_w = iter->end_x;
    iter->current_tile_h = iter->end_y;
    iter->current_tile_d = iter->end_z;
    
    iter->cpu_ptr = tile->region->cpu_ptr;
    iter->gpu_ptr = tile->region->gpu_ptr;
    
    iter->last_tile = (tile->num_tiles <= 1);
    
    return NEARMEM_OK;
}

/*
 * Advance to next tile
 */
bool nm_tile_next(nm_tile_iterator_t *iter)
{
    if (!iter || iter->last_tile)
        return false;
    
    /* Writeback current tile if dirty */
    if (iter->dirty) {
        if (iter->ndim == 2) {
            nm_tile_2d_t *tile = iter->tile_2d;
            tile_writeback_2d(tile, iter, 
                             tile->prefetch_buffer[tile->active_buffer]);
        }
        iter->dirty = false;
    }
    
    /* Advance position */
    iter->tile_idx++;
    
    if (iter->ndim == 2) {
        nm_tile_2d_t *tile = iter->tile_2d;
        
        /* Update tile coordinates */
        iter->tile_x++;
        if (iter->tile_x >= tile->num_tiles_x) {
            iter->tile_x = 0;
            iter->tile_y++;
        }
        
        /* Calculate new bounds */
        iter->start_x = iter->tile_x * tile->tile_w;
        iter->start_y = iter->tile_y * tile->tile_h;
        iter->end_x = iter->start_x + tile->tile_w;
        iter->end_y = iter->start_y + tile->tile_h;
        
        if (iter->end_x > tile->width) iter->end_x = tile->width;
        if (iter->end_y > tile->height) iter->end_y = tile->height;
        
        iter->current_tile_w = iter->end_x - iter->start_x;
        iter->current_tile_h = iter->end_y - iter->start_y;
        
        /* Swap buffers (double-buffering) */
        tile->active_buffer = 1 - tile->active_buffer;
        
        /* Prefetch next tile async */
        if (iter->tile_idx + 1 < tile->num_tiles && tile->prefetch_depth > 0) {
            int next_buffer = 1 - tile->active_buffer;
            tile_prefetch_2d(tile, iter, iter->tile_idx + 1,
                            tile->prefetch_buffer[next_buffer]);
        }
        
        iter->last_tile = (iter->tile_idx >= tile->num_tiles - 1);
    }
    else if (iter->ndim == 1) {
        nm_tile_1d_t *tile = iter->tile_1d;
        
        iter->start_x = iter->tile_idx * tile->tile_size;
        iter->end_x = iter->start_x + tile->tile_size;
        if (iter->end_x > tile->length) iter->end_x = tile->length;
        iter->current_tile_w = iter->end_x - iter->start_x;
        
        iter->last_tile = (iter->tile_idx >= tile->num_tiles - 1);
    }
    else if (iter->ndim == 3) {
        nm_tile_3d_t *tile = iter->tile_3d;
        
        iter->tile_x++;
        if (iter->tile_x >= tile->num_tiles_x) {
            iter->tile_x = 0;
            iter->tile_y++;
            if (iter->tile_y >= tile->num_tiles_y) {
                iter->tile_y = 0;
                iter->tile_z++;
            }
        }
        
        iter->start_x = iter->tile_x * tile->tile_w;
        iter->start_y = iter->tile_y * tile->tile_h;
        iter->start_z = iter->tile_z * tile->tile_d;
        iter->end_x = iter->start_x + tile->tile_w;
        iter->end_y = iter->start_y + tile->tile_h;
        iter->end_z = iter->start_z + tile->tile_d;
        
        if (iter->end_x > tile->width) iter->end_x = tile->width;
        if (iter->end_y > tile->height) iter->end_y = tile->height;
        if (iter->end_z > tile->depth) iter->end_z = tile->depth;
        
        iter->current_tile_w = iter->end_x - iter->start_x;
        iter->current_tile_h = iter->end_y - iter->start_y;
        iter->current_tile_d = iter->end_z - iter->start_z;
        
        iter->last_tile = (iter->tile_idx >= tile->num_tiles - 1);
    }
    
    return true;
}

/*
 * End iteration
 */
void nm_tile_end(nm_tile_iterator_t *iter)
{
    if (!iter)
        return;
    
    /* Flush any pending writes */
    if (iter->dirty) {
        if (iter->ndim == 2) {
            nm_tile_2d_t *tile = iter->tile_2d;
            tile_writeback_2d(tile, iter,
                             tile->prefetch_buffer[tile->active_buffer]);
        }
        iter->dirty = false;
    }
    
    /* Synchronize */
    if (iter->ndim == 2 && iter->tile_2d->stream && cuStreamSynchronize) {
        cuStreamSynchronize(iter->tile_2d->stream);
    }
}

/*
 * Data access
 */
void *nm_tile_get_cpu(nm_tile_iterator_t *iter)
{
    if (!iter)
        return NULL;
    
    /* Return direct pointer into VRAM (via BAR1 mmap) */
    if (iter->ndim == 2) {
        nm_tile_2d_t *tile = iter->tile_2d;
        size_t offset = iter->start_y * tile->pitch + 
                        iter->start_x * tile->element_size;
        return (char*)tile->region->cpu_ptr + offset;
    }
    else if (iter->ndim == 1) {
        nm_tile_1d_t *tile = iter->tile_1d;
        size_t offset = iter->start_x * tile->element_size;
        return (char*)tile->region->cpu_ptr + offset;
    }
    
    return NULL;
}

void *nm_tile_get_gpu(nm_tile_iterator_t *iter)
{
    if (!iter)
        return NULL;
    
    /* Return device pointer (offset in VRAM) */
    if (iter->ndim == 2) {
        nm_tile_2d_t *tile = iter->tile_2d;
        size_t offset = iter->start_y * tile->pitch + 
                        iter->start_x * tile->element_size;
        return (char*)tile->region->gpu_ptr + offset;
    }
    else if (iter->ndim == 1) {
        nm_tile_1d_t *tile = iter->tile_1d;
        size_t offset = iter->start_x * tile->element_size;
        return (char*)tile->region->gpu_ptr + offset;
    }
    
    return NULL;
}

void *nm_tile_get_shared(nm_tile_iterator_t *iter)
{
    if (!iter || !iter->prefetched)
        return NULL;
    
    /* Return pointer to prefetch buffer */
    if (iter->ndim == 2) {
        return iter->tile_2d->prefetch_buffer[iter->tile_2d->active_buffer];
    }
    else if (iter->ndim == 1) {
        return iter->tile_1d->prefetch_buffer[iter->tile_1d->active_buffer];
    }
    
    return NULL;
}

void nm_tile_mark_dirty(nm_tile_iterator_t *iter)
{
    if (iter)
        iter->dirty = true;
}

void nm_tile_prefetch(nm_tile_iterator_t *iter, size_t tile_idx)
{
    if (!iter || iter->ndim != 2)
        return;
    
    nm_tile_2d_t *tile = iter->tile_2d;
    if (tile_idx >= tile->num_tiles)
        return;
    
    int buffer = 1 - tile->active_buffer;
    tile_prefetch_2d(tile, iter, tile_idx, tile->prefetch_buffer[buffer]);
}

/*
 * Synchronization
 */
void nm_tile_sync_cpu(nm_tile_iterator_t *iter)
{
    if (!iter)
        return;
    
    nearmem_ctx_t *ctx = NULL;
    if (iter->ndim == 2) ctx = iter->tile_2d->ctx;
    else if (iter->ndim == 1) ctx = iter->tile_1d->ctx;
    else if (iter->ndim == 3) ctx = iter->tile_3d->ctx;
    
    if (ctx)
        nearmem_sync(ctx, NEARMEM_SYNC_CPU_TO_GPU);
}

void nm_tile_sync_gpu(nm_tile_iterator_t *iter)
{
    if (!iter)
        return;
    
    nearmem_ctx_t *ctx = NULL;
    if (iter->ndim == 2) ctx = iter->tile_2d->ctx;
    else if (iter->ndim == 1) ctx = iter->tile_1d->ctx;
    else if (iter->ndim == 3) ctx = iter->tile_3d->ctx;
    
    if (ctx)
        nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
}

void nm_tile_barrier(nm_tile_iterator_t *iter)
{
    nm_tile_sync_cpu(iter);
    nm_tile_sync_gpu(iter);
}

/*
 * Statistics
 */
void nm_tile_get_stats(nm_tile_iterator_t *iter, nm_tile_stats_t *stats)
{
    if (!iter || !stats)
        return;
    
    memset(stats, 0, sizeof(*stats));
    
    stats->tiles_processed = iter->tile_idx + 1;
    stats->total_prefetch_us = iter->prefetch_time_us;
    stats->total_compute_us = iter->compute_time_us;
    stats->total_writeback_us = iter->writeback_time_us;
    
    /* Calculate bandwidths */
    if (iter->ndim == 2) {
        nm_tile_2d_t *tile = iter->tile_2d;
        size_t tile_bytes = tile->tile_w * tile->tile_h * tile->element_size;
        
        stats->bytes_prefetched = stats->tiles_processed * tile_bytes;
        stats->bytes_written_back = stats->tiles_processed * tile_bytes;  /* Approx */
        
        if (stats->total_prefetch_us > 0) {
            stats->prefetch_bandwidth_gbps = 
                (stats->bytes_prefetched / 1e9) / (stats->total_prefetch_us / 1e6);
        }
        if (stats->total_writeback_us > 0) {
            stats->writeback_bandwidth_gbps =
                (stats->bytes_written_back / 1e9) / (stats->total_writeback_us / 1e6);
        }
    }
}

void nm_tile_print_stats(nm_tile_stats_t *stats)
{
    if (!stats)
        return;
    
    printf("Tile Statistics:\n");
    printf("  Tiles processed:      %zu\n", stats->tiles_processed);
    printf("  Bytes prefetched:     %.2f MB\n", stats->bytes_prefetched / 1e6);
    printf("  Bytes written back:   %.2f MB\n", stats->bytes_written_back / 1e6);
    printf("  Prefetch time:        %.2f ms\n", stats->total_prefetch_us / 1000.0);
    printf("  Compute time:         %.2f ms\n", stats->total_compute_us / 1000.0);
    printf("  Writeback time:       %.2f ms\n", stats->total_writeback_us / 1000.0);
    printf("  Prefetch bandwidth:   %.2f GB/s\n", stats->prefetch_bandwidth_gbps);
    printf("  Writeback bandwidth:  %.2f GB/s\n", stats->writeback_bandwidth_gbps);
}
