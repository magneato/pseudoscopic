/* SPDX-License-Identifier: MIT */
/*
 * nmc.c - Near-Memory Computing Library Implementation
 *
 * The bridge between pseudoscopic and CUDA.
 * Data lives in VRAM. We just visit.
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <dirent.h>
#include <pthread.h>
#include <time.h>

#include "nmc.h"

/* CUDA headers - will be dynamically loaded or linked */
#include <cuda_runtime.h>

/*
 * Internal Structures
 */

struct nmc_region {
    nmc_context_t   *ctx;           /* Parent context */
    void            *cpu_ptr;       /* WC-mapped CPU pointer */
    void            *gpu_ptr;       /* CUDA device pointer */
    size_t          size;           /* Region size */
    size_t          offset;         /* Offset in VRAM */
    nmc_mem_flags_t flags;          /* Memory flags */
    bool            mapped;         /* CPU mapping active */
    struct nmc_region *next;        /* Linked list */
};

struct nmc_stream {
    nmc_context_t   *ctx;
    cudaStream_t    cuda_stream;
    bool            busy;
};

struct nmc_context {
    /* Device handles */
    int             ps_fd;          /* Pseudoscopic block device fd */
    int             cuda_device;    /* CUDA device index */
    
    /* Memory mapping */
    void            *vram_base;     /* mmap'd VRAM base */
    size_t          vram_size;      /* Total VRAM size */
    size_t          vram_used;      /* Allocated bytes */
    
    /* Region tracking */
    nmc_region_t    *regions;       /* Allocated regions list */
    pthread_mutex_t region_lock;    /* Protects region list */
    
    /* Allocation tracking (simple bump allocator for now) */
    size_t          alloc_offset;   /* Next free offset */
    
    /* Statistics */
    nmc_stats_t     stats;
    pthread_mutex_t stats_lock;
    
    /* CUDA stream for sync operations */
    cudaStream_t    default_stream;
};

/*
 * Helper: Get current time in milliseconds
 */
static double get_time_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1000000.0;
}

/*
 * Helper: Round up to page boundary
 */
static size_t align_to_page(size_t size)
{
    size_t page_size = sysconf(_SC_PAGESIZE);
    return (size + page_size - 1) & ~(page_size - 1);
}

/*
 * Helper: Find pseudoscopic device
 */
static int find_pseudoscopic_device(char *path, size_t path_len)
{
    /* Try ramdisk devices first */
    for (int i = 0; i < 8; i++) {
        snprintf(path, path_len, "/dev/psdisk%d", i);
        if (access(path, R_OK | W_OK) == 0)
            return 0;
    }
    
    /* Try swap devices */
    for (int i = 0; i < 8; i++) {
        snprintf(path, path_len, "/dev/pswap%d", i);
        if (access(path, R_OK | W_OK) == 0)
            return 0;
    }
    
    return -1;
}

/*
 * Helper: Get device size
 */
static size_t get_device_size(int fd)
{
    off_t size = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    return (size_t)size;
}

/*
 * Helper: Match CUDA device to pseudoscopic device
 */
static int match_cuda_device(int ps_fd)
{
    int device_count;
    cudaError_t err;
    
    err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0)
        return -1;
    
    /* For now, just return the first non-display device */
    /* TODO: Match by PCI bus ID from sysfs */
    for (int i = 0; i < device_count; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        
        /* Skip devices in TCC mode (display) if possible */
        /* Actually, for our use case any CUDA device works */
        return i;
    }
    
    return 0;
}

/*
 * nmc_init - Initialize NMC context
 */
nmc_error_t nmc_init(const char *device_path, 
                     int cuda_device,
                     nmc_context_t **ctx_out)
{
    nmc_context_t *ctx;
    char path[256];
    cudaError_t cuda_err;
    
    if (!ctx_out)
        return NMC_ERROR_INVALID_ARG;
    
    ctx = calloc(1, sizeof(*ctx));
    if (!ctx)
        return NMC_ERROR_OUT_OF_MEMORY;
    
    pthread_mutex_init(&ctx->region_lock, NULL);
    pthread_mutex_init(&ctx->stats_lock, NULL);
    
    /* Find or use specified device */
    if (device_path) {
        strncpy(path, device_path, sizeof(path) - 1);
    } else {
        if (find_pseudoscopic_device(path, sizeof(path)) < 0) {
            free(ctx);
            return NMC_ERROR_NO_DEVICE;
        }
    }
    
    /* Open pseudoscopic device */
    ctx->ps_fd = open(path, O_RDWR);
    if (ctx->ps_fd < 0) {
        free(ctx);
        return NMC_ERROR_NO_DEVICE;
    }
    
    /* Get VRAM size */
    ctx->vram_size = get_device_size(ctx->ps_fd);
    if (ctx->vram_size == 0) {
        close(ctx->ps_fd);
        free(ctx);
        return NMC_ERROR_NO_DEVICE;
    }
    
    /* mmap entire VRAM for CPU access */
    ctx->vram_base = mmap(NULL, ctx->vram_size,
                          PROT_READ | PROT_WRITE,
                          MAP_SHARED,
                          ctx->ps_fd, 0);
    if (ctx->vram_base == MAP_FAILED) {
        close(ctx->ps_fd);
        free(ctx);
        return NMC_ERROR_MMAP_FAILED;
    }
    
    /* Initialize CUDA */
    if (cuda_device < 0) {
        cuda_device = match_cuda_device(ctx->ps_fd);
    }
    
    cuda_err = cudaSetDevice(cuda_device);
    if (cuda_err != cudaSuccess) {
        munmap(ctx->vram_base, ctx->vram_size);
        close(ctx->ps_fd);
        free(ctx);
        return NMC_ERROR_NO_CUDA;
    }
    
    ctx->cuda_device = cuda_device;
    
    /* Create default CUDA stream */
    cuda_err = cudaStreamCreate(&ctx->default_stream);
    if (cuda_err != cudaSuccess) {
        munmap(ctx->vram_base, ctx->vram_size);
        close(ctx->ps_fd);
        free(ctx);
        return NMC_ERROR_CUDA_FAILED;
    }
    
    /* Log initialization */
    fprintf(stderr, "NMC: Initialized with %zu MB VRAM on CUDA device %d\n",
            ctx->vram_size >> 20, cuda_device);
    
    *ctx_out = ctx;
    return NMC_SUCCESS;
}

/*
 * nmc_shutdown - Destroy NMC context
 */
void nmc_shutdown(nmc_context_t *ctx)
{
    if (!ctx)
        return;
    
    /* Free all regions */
    pthread_mutex_lock(&ctx->region_lock);
    while (ctx->regions) {
        nmc_region_t *r = ctx->regions;
        ctx->regions = r->next;
        free(r);
    }
    pthread_mutex_unlock(&ctx->region_lock);
    
    /* Cleanup CUDA */
    cudaStreamDestroy(ctx->default_stream);
    
    /* Cleanup mmap */
    if (ctx->vram_base && ctx->vram_base != MAP_FAILED) {
        munmap(ctx->vram_base, ctx->vram_size);
    }
    
    /* Close device */
    if (ctx->ps_fd >= 0) {
        close(ctx->ps_fd);
    }
    
    pthread_mutex_destroy(&ctx->region_lock);
    pthread_mutex_destroy(&ctx->stats_lock);
    
    free(ctx);
}

/*
 * nmc_get_stats / nmc_reset_stats
 */
nmc_error_t nmc_get_stats(nmc_context_t *ctx, nmc_stats_t *stats)
{
    if (!ctx || !stats)
        return NMC_ERROR_INVALID_ARG;
    
    pthread_mutex_lock(&ctx->stats_lock);
    *stats = ctx->stats;
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return NMC_SUCCESS;
}

void nmc_reset_stats(nmc_context_t *ctx)
{
    if (!ctx)
        return;
    
    pthread_mutex_lock(&ctx->stats_lock);
    memset(&ctx->stats, 0, sizeof(ctx->stats));
    pthread_mutex_unlock(&ctx->stats_lock);
}

size_t nmc_get_capacity(nmc_context_t *ctx)
{
    return ctx ? ctx->vram_size : 0;
}

size_t nmc_get_available(nmc_context_t *ctx)
{
    return ctx ? ctx->vram_size - ctx->vram_used : 0;
}

/*
 * nmc_alloc - Allocate VRAM region
 */
nmc_error_t nmc_alloc(nmc_context_t *ctx,
                      size_t size,
                      nmc_mem_flags_t flags,
                      nmc_region_t **region_out)
{
    nmc_region_t *region;
    size_t aligned_size;
    
    if (!ctx || !region_out || size == 0)
        return NMC_ERROR_INVALID_ARG;
    
    aligned_size = align_to_page(size);
    
    /* Check capacity */
    if (ctx->alloc_offset + aligned_size > ctx->vram_size)
        return NMC_ERROR_OUT_OF_MEMORY;
    
    region = calloc(1, sizeof(*region));
    if (!region)
        return NMC_ERROR_OUT_OF_MEMORY;
    
    region->ctx = ctx;
    region->size = aligned_size;
    region->flags = flags;
    region->offset = ctx->alloc_offset;
    
    /* CPU pointer via mmap'd region */
    region->cpu_ptr = (char *)ctx->vram_base + region->offset;
    region->mapped = true;
    
    /*
     * GPU pointer: This is the key insight.
     *
     * The pseudoscopic mmap gives us CPU access to VRAM via BAR1.
     * The GPU sees the same VRAM as its device memory.
     *
     * We use cuMemHostRegister or a direct device pointer.
     * For simplicity, we'll use the device base + offset approach.
     *
     * In practice, we'd query the GPU's view of VRAM and calculate
     * the matching device pointer.
     */
    
    /* For now, use CUDA managed memory as a bridge */
    /* In production, this would use nvidia-peermem or UVM */
    cudaError_t err = cudaMalloc(&region->gpu_ptr, aligned_size);
    if (err != cudaSuccess) {
        free(region);
        return NMC_ERROR_CUDA_FAILED;
    }
    
    /* Update allocator state */
    ctx->alloc_offset += aligned_size;
    ctx->vram_used += aligned_size;
    
    /* Add to region list */
    pthread_mutex_lock(&ctx->region_lock);
    region->next = ctx->regions;
    ctx->regions = region;
    pthread_mutex_unlock(&ctx->region_lock);
    
    *region_out = region;
    return NMC_SUCCESS;
}

void nmc_free(nmc_region_t *region)
{
    nmc_context_t *ctx;
    nmc_region_t **pp;
    
    if (!region)
        return;
    
    ctx = region->ctx;
    
    /* Remove from list */
    pthread_mutex_lock(&ctx->region_lock);
    for (pp = &ctx->regions; *pp; pp = &(*pp)->next) {
        if (*pp == region) {
            *pp = region->next;
            break;
        }
    }
    pthread_mutex_unlock(&ctx->region_lock);
    
    /* Free CUDA memory */
    if (region->gpu_ptr) {
        cudaFree(region->gpu_ptr);
    }
    
    ctx->vram_used -= region->size;
    
    /* Note: We don't actually free the VRAM offset - that would
     * require a more sophisticated allocator. For now, regions
     * are freed but space isn't reclaimed until context shutdown. */
    
    free(region);
}

void *nmc_cpu_ptr(nmc_region_t *region)
{
    return region ? region->cpu_ptr : NULL;
}

void *nmc_gpu_ptr(nmc_region_t *region)
{
    return region ? region->gpu_ptr : NULL;
}

size_t nmc_size(nmc_region_t *region)
{
    return region ? region->size : 0;
}

/*
 * nmc_load - Load data into VRAM
 */
nmc_error_t nmc_load(nmc_region_t *region,
                     size_t offset,
                     const void *src,
                     size_t size)
{
    nmc_context_t *ctx;
    double start_time;
    
    if (!region || !src)
        return NMC_ERROR_INVALID_ARG;
    
    if (offset + size > region->size)
        return NMC_ERROR_INVALID_ARG;
    
    ctx = region->ctx;
    start_time = get_time_ms();
    
    /*
     * Write to VRAM via write-combining mapped pointer.
     * This goes directly to GPU memory via BAR1.
     *
     * For optimal performance, use our ASM memcpy_wc routines.
     * For portability, we use standard memcpy here.
     */
    memcpy((char *)region->cpu_ptr + offset, src, size);
    
    /* Also update CUDA view */
    cudaMemcpy((char *)region->gpu_ptr + offset, src, size, 
               cudaMemcpyHostToDevice);
    
    /* Update stats */
    pthread_mutex_lock(&ctx->stats_lock);
    ctx->stats.bytes_to_vram += size;
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return NMC_SUCCESS;
}

/*
 * nmc_extract - Read data from VRAM
 */
nmc_error_t nmc_extract(nmc_region_t *region,
                        size_t offset,
                        void *dst,
                        size_t size)
{
    nmc_context_t *ctx;
    
    if (!region || !dst)
        return NMC_ERROR_INVALID_ARG;
    
    if (offset + size > region->size)
        return NMC_ERROR_INVALID_ARG;
    
    ctx = region->ctx;
    
    /* Read via CUDA for coherent view */
    cudaMemcpy(dst, (char *)region->gpu_ptr + offset, size,
               cudaMemcpyDeviceToHost);
    
    /* Update stats */
    pthread_mutex_lock(&ctx->stats_lock);
    ctx->stats.bytes_from_vram += size;
    ctx->stats.cpu_touches++;
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return NMC_SUCCESS;
}

/*
 * nmc_sync - Synchronize CPU and GPU
 */
nmc_error_t nmc_sync(nmc_context_t *ctx, nmc_sync_mode_t mode)
{
    double start_time;
    
    if (!ctx)
        return NMC_ERROR_INVALID_ARG;
    
    start_time = get_time_ms();
    
    switch (mode) {
    case NMC_SYNC_NONE:
        break;
        
    case NMC_SYNC_CPU_TO_GPU:
        /* Memory fence to ensure CPU writes visible */
        __sync_synchronize();
        break;
        
    case NMC_SYNC_GPU_TO_CPU:
        /* Wait for GPU operations */
        cudaDeviceSynchronize();
        break;
        
    case NMC_SYNC_FULL:
        __sync_synchronize();
        cudaDeviceSynchronize();
        __sync_synchronize();
        break;
    }
    
    pthread_mutex_lock(&ctx->stats_lock);
    ctx->stats.sync_count++;
    ctx->stats.time_in_sync_ms += get_time_ms() - start_time;
    pthread_mutex_unlock(&ctx->stats_lock);
    
    return NMC_SUCCESS;
}

void nmc_fence_cpu(nmc_region_t *region)
{
    __sync_synchronize();
    if (region) {
        pthread_mutex_lock(&region->ctx->stats_lock);
        region->ctx->stats.sync_count++;
        pthread_mutex_unlock(&region->ctx->stats_lock);
    }
}

nmc_error_t nmc_fence_gpu(nmc_context_t *ctx)
{
    if (!ctx)
        return NMC_ERROR_INVALID_ARG;
    
    cudaStreamSynchronize(ctx->default_stream);
    return NMC_SUCCESS;
}

/*
 * nmc_stream_create / destroy / sync
 */
nmc_error_t nmc_stream_create(nmc_context_t *ctx, nmc_stream_t **stream_out)
{
    nmc_stream_t *stream;
    cudaError_t err;
    
    if (!ctx || !stream_out)
        return NMC_ERROR_INVALID_ARG;
    
    stream = calloc(1, sizeof(*stream));
    if (!stream)
        return NMC_ERROR_OUT_OF_MEMORY;
    
    stream->ctx = ctx;
    
    err = cudaStreamCreate(&stream->cuda_stream);
    if (err != cudaSuccess) {
        free(stream);
        return NMC_ERROR_CUDA_FAILED;
    }
    
    *stream_out = stream;
    return NMC_SUCCESS;
}

void nmc_stream_destroy(nmc_stream_t *stream)
{
    if (!stream)
        return;
    
    cudaStreamDestroy(stream->cuda_stream);
    free(stream);
}

nmc_error_t nmc_stream_sync(nmc_stream_t *stream)
{
    if (!stream)
        return NMC_ERROR_INVALID_ARG;
    
    cudaStreamSynchronize(stream->cuda_stream);
    return NMC_SUCCESS;
}

/*
 * nmc_memset - GPU-accelerated memset
 */
nmc_error_t nmc_memset(nmc_region_t *region,
                       size_t offset,
                       int value,
                       size_t size,
                       nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    double start_time;
    
    if (!region)
        return NMC_ERROR_INVALID_ARG;
    
    if (offset + size > region->size)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : region->ctx->default_stream;
    start_time = get_time_ms();
    
    cudaMemsetAsync((char *)region->gpu_ptr + offset, value, size, cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    pthread_mutex_lock(&region->ctx->stats_lock);
    region->ctx->stats.gpu_ops++;
    region->ctx->stats.pcie_avoided += size; /* Would have been a transfer */
    region->ctx->stats.time_in_gpu_ms += get_time_ms() - start_time;
    pthread_mutex_unlock(&region->ctx->stats_lock);
    
    return NMC_SUCCESS;
}

/*
 * nmc_memcpy_internal - Copy within VRAM
 */
nmc_error_t nmc_memcpy_internal(nmc_region_t *dst,
                                size_t dst_offset,
                                nmc_region_t *src,
                                size_t src_offset,
                                size_t size,
                                nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    double start_time;
    
    if (!dst || !src)
        return NMC_ERROR_INVALID_ARG;
    
    if (dst_offset + size > dst->size || src_offset + size > src->size)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : dst->ctx->default_stream;
    start_time = get_time_ms();
    
    cudaMemcpyAsync((char *)dst->gpu_ptr + dst_offset,
                    (char *)src->gpu_ptr + src_offset,
                    size,
                    cudaMemcpyDeviceToDevice,
                    cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    pthread_mutex_lock(&dst->ctx->stats_lock);
    dst->ctx->stats.gpu_ops++;
    dst->ctx->stats.pcie_avoided += size * 2; /* Would have been D→H→D */
    dst->ctx->stats.time_in_gpu_ms += get_time_ms() - start_time;
    pthread_mutex_unlock(&dst->ctx->stats_lock);
    
    return NMC_SUCCESS;
}

/*
 * nmc_error_string - Get error description
 */
const char *nmc_error_string(nmc_error_t error)
{
    switch (error) {
    case NMC_SUCCESS:           return "Success";
    case NMC_ERROR_NO_DEVICE:   return "Pseudoscopic device not found";
    case NMC_ERROR_NO_CUDA:     return "CUDA runtime not available";
    case NMC_ERROR_MMAP_FAILED: return "Failed to mmap VRAM";
    case NMC_ERROR_CUDA_FAILED: return "CUDA operation failed";
    case NMC_ERROR_OUT_OF_MEMORY: return "Out of VRAM";
    case NMC_ERROR_INVALID_ARG: return "Invalid argument";
    case NMC_ERROR_SYNC_TIMEOUT: return "Synchronization timeout";
    case NMC_ERROR_NOT_ALIGNED: return "Not properly aligned";
    case NMC_ERROR_BUSY:        return "Resource busy";
    default:                    return "Unknown error";
    }
}

/*
 * nmc_print_stats - Print statistics
 */
void nmc_print_stats(nmc_context_t *ctx, FILE *fp)
{
    nmc_stats_t stats;
    
    if (!ctx)
        return;
    
    if (!fp)
        fp = stderr;
    
    nmc_get_stats(ctx, &stats);
    
    fprintf(fp, "\n");
    fprintf(fp, "╔══════════════════════════════════════════════╗\n");
    fprintf(fp, "║     Near-Memory Computing Statistics         ║\n");
    fprintf(fp, "╠══════════════════════════════════════════════╣\n");
    fprintf(fp, "║ Bytes to VRAM:     %15lu           ║\n", stats.bytes_to_vram);
    fprintf(fp, "║ Bytes from VRAM:   %15lu           ║\n", stats.bytes_from_vram);
    fprintf(fp, "║ GPU operations:    %15lu           ║\n", stats.gpu_ops);
    fprintf(fp, "║ CPU touches:       %15lu           ║\n", stats.cpu_touches);
    fprintf(fp, "║ Sync barriers:     %15lu           ║\n", stats.sync_count);
    fprintf(fp, "║ PCIe bytes avoided:%15lu           ║\n", stats.pcie_avoided);
    fprintf(fp, "║ GPU time:          %12.2f ms           ║\n", stats.time_in_gpu_ms);
    fprintf(fp, "║ Sync time:         %12.2f ms           ║\n", stats.time_in_sync_ms);
    fprintf(fp, "╠══════════════════════════════════════════════╣\n");
    
    if (stats.bytes_to_vram + stats.bytes_from_vram > 0) {
        double efficiency = (double)stats.pcie_avoided / 
                           (stats.bytes_to_vram + stats.bytes_from_vram + stats.pcie_avoided);
        fprintf(fp, "║ PCIe Efficiency:   %12.1f%%             ║\n", efficiency * 100);
    }
    
    fprintf(fp, "╚══════════════════════════════════════════════╝\n");
    fprintf(fp, "\n");
}
