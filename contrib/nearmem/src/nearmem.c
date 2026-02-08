/*
 * nearmem.c - Near-Memory Computing Implementation
 *
 * The magic trick: CPU writes to BAR1, GPU reads from VRAM.
 * Same physical memory. Different access paths. No copy.
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <linux/fs.h>
#include <dirent.h>
#include <errno.h>

#include "nearmem.h"

/* CUDA headers - optional, fallback to dlopen */
#ifdef NEARMEM_USE_CUDA_HEADERS
#include <cuda.h>
#include <cuda_runtime.h>
#else
/* Dynamic loading of CUDA */
#include <dlfcn.h>
typedef void* CUcontext;
typedef void* CUdeviceptr;
typedef void* CUstream;
typedef int CUdevice;
typedef int CUresult;

#define CUDA_SUCCESS 0

/* Function pointers loaded at runtime */
static CUresult (*cuInit)(unsigned int);
static CUresult (*cuDeviceGet)(CUdevice*, int);
static CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
static CUresult (*cuCtxDestroy)(CUcontext);
static CUresult (*cuCtxSynchronize)(void);
static CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
static CUresult (*cuMemFree)(CUdeviceptr);
static CUresult (*cuMemsetD8)(CUdeviceptr, unsigned char, size_t);
static CUresult (*cuMemcpyDtoD)(CUdeviceptr, CUdeviceptr, size_t);
static CUresult (*cuStreamCreate)(CUstream*, unsigned int);
static CUresult (*cuStreamDestroy)(CUstream);
static CUresult (*cuStreamSynchronize)(CUstream);
static CUresult (*cuLaunchKernel)(void*, unsigned int, unsigned int, unsigned int,
                                   unsigned int, unsigned int, unsigned int,
                                   unsigned int, CUstream, void**, void**);

static void *cuda_lib = NULL;
#endif

/* Internal state for stats tracking */
typedef struct {
    uint64_t bytes_to_gpu;
    uint64_t bytes_from_gpu;
    uint64_t gpu_operations;
    uint64_t gpu_time_us;
} nearmem_stats_t;

/* container_of macro for safe casting from public to internal context */
#define container_of(ptr, type, member) \
    ((type *)((char *)(ptr) - offsetof(type, member)))

/* Extended context with internal state */
typedef struct {
    nearmem_ctx_t   public;     /* Must be first member for direct cast */
    nearmem_stats_t stats;
    uint64_t        alloc_offset;   /* Simple bump allocator */
} nearmem_ctx_internal_t;

/* Safe accessor for internal context from public context */
static inline nearmem_ctx_internal_t *get_internal_ctx(nearmem_ctx_t *ctx) {
    /* Since 'public' is first member, we can directly cast */
    return (nearmem_ctx_internal_t *)ctx;
}


/*
 * Error strings
 */
static const char *error_strings[] = {
    [0] = "Success",
    [1] = "Initialization failed",
    [2] = "No device found",
    [3] = "Memory mapping failed",
    [4] = "CUDA error",
    [5] = "Synchronization error",
    [6] = "Bounds check failed",
    [7] = "Invalid argument",
};

const char *nearmem_strerror(nearmem_error_t err)
{
    int idx = -err;
    if (idx < 0 || idx >= (int)(sizeof(error_strings)/sizeof(error_strings[0])))
        return "Unknown error";
    return error_strings[idx];
}

/*
 * Load CUDA library dynamically
 */
#ifndef NEARMEM_USE_CUDA_HEADERS
static int load_cuda(void)
{
    if (cuda_lib)
        return 0;
    
    cuda_lib = dlopen("libcuda.so.1", RTLD_NOW);
    if (!cuda_lib) {
        cuda_lib = dlopen("libcuda.so", RTLD_NOW);
        if (!cuda_lib) {
            fprintf(stderr, "nearmem: failed to load libcuda.so: %s\n", dlerror());
            return -1;
        }
    }
    
    #define LOAD_SYM(name) do { \
        *(void**)(&name) = dlsym(cuda_lib, #name); \
        if (!name) { \
            fprintf(stderr, "nearmem: failed to load %s\n", #name); \
            return -1; \
        } \
    } while(0)
    
    LOAD_SYM(cuInit);
    LOAD_SYM(cuDeviceGet);
    LOAD_SYM(cuCtxCreate);
    LOAD_SYM(cuCtxDestroy);
    LOAD_SYM(cuCtxSynchronize);
    LOAD_SYM(cuMemAlloc);
    LOAD_SYM(cuMemFree);
    LOAD_SYM(cuMemsetD8);
    LOAD_SYM(cuMemcpyDtoD);
    LOAD_SYM(cuStreamCreate);
    LOAD_SYM(cuStreamDestroy);
    LOAD_SYM(cuStreamSynchronize);
    LOAD_SYM(cuLaunchKernel);
    
    #undef LOAD_SYM
    
    return 0;
}
#endif

/*
 * Find pseudoscopic block device
 */
static int find_psdisk(char *path, size_t path_len)
{
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir("/dev");
    if (!dir)
        return -1;
    
    while ((entry = readdir(dir)) != NULL) {
        if (strncmp(entry->d_name, "psdisk", 6) == 0 ||
            strncmp(entry->d_name, "pswap", 5) == 0) {
            snprintf(path, path_len, "/dev/%s", entry->d_name);
            closedir(dir);
            return 0;
        }
    }
    
    closedir(dir);
    return -1;
}

/*
 * Get device size via ioctl
 */
static size_t get_device_size(int fd)
{
    uint64_t size = 0;
    
    if (ioctl(fd, BLKGETSIZE64, &size) == 0)
        return (size_t)size;
    
    /* Fallback: seek to end */
    off_t end = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    
    return (end > 0) ? (size_t)end : 0;
}

/*
 * Initialize context
 */
nearmem_error_t nearmem_init(nearmem_ctx_t *ctx,
                              const char *device_path,
                              int cuda_device)
{
    nearmem_ctx_internal_t *ictx;
    CUdevice cu_dev;
    CUresult cu_err;
    int fd;
    void *mapped;
    size_t size;
    
    if (!ctx || !device_path)
        return NEARMEM_ERROR_INVALID;
    
    /* Allocate internal context */
    ictx = calloc(1, sizeof(*ictx));
    if (!ictx)
        return NEARMEM_ERROR_INIT;
    
    /* Copy public portion back */
    memset(ctx, 0, sizeof(*ctx));
    
#ifndef NEARMEM_USE_CUDA_HEADERS
    /* Load CUDA library */
    if (load_cuda() != 0) {
        free(ictx);
        return NEARMEM_ERROR_CUDA;
    }
#endif
    
    /* Initialize CUDA */
    cu_err = cuInit(0);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nearmem: cuInit failed: %d\n", cu_err);
        free(ictx);
        return NEARMEM_ERROR_CUDA;
    }
    
    /* Get CUDA device */
    cu_err = cuDeviceGet(&cu_dev, cuda_device);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nearmem: cuDeviceGet(%d) failed: %d\n", cuda_device, cu_err);
        free(ictx);
        return NEARMEM_ERROR_CUDA;
    }
    
    /* Create CUDA context */
    cu_err = cuCtxCreate((CUcontext*)&ctx->cuda_ctx, 0, cu_dev);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nearmem: cuCtxCreate failed: %d\n", cu_err);
        free(ictx);
        return NEARMEM_ERROR_CUDA;
    }
    
    /* Open pseudoscopic block device */
    fd = open(device_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "nearmem: failed to open %s: %s\n", 
                device_path, strerror(errno));
        cuCtxDestroy(ctx->cuda_ctx);
        free(ictx);
        return NEARMEM_ERROR_NO_DEVICE;
    }
    
    /* Get device size */
    size = get_device_size(fd);
    if (size == 0) {
        fprintf(stderr, "nearmem: failed to get device size\n");
        close(fd);
        cuCtxDestroy(ctx->cuda_ctx);
        free(ictx);
        return NEARMEM_ERROR_NO_DEVICE;
    }
    
    /*
     * mmap the entire device
     *
     * This maps the pseudoscopic block device into our address space.
     * Since pseudoscopic exposes VRAM via BAR1, this gives us direct
     * CPU access to GPU memory.
     *
     * MAP_SHARED: Writes are visible to other mappings (including GPU)
     * MAP_NORESERVE: Don't reserve swap (it's device memory)
     */
    mapped = mmap(NULL, size, PROT_READ | PROT_WRITE, 
                  MAP_SHARED | MAP_NORESERVE, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "nearmem: mmap failed: %s\n", strerror(errno));
        close(fd);
        cuCtxDestroy(ctx->cuda_ctx);
        free(ictx);
        return NEARMEM_ERROR_MMAP;
    }
    
    /* Advise kernel about access pattern */
    madvise(mapped, size, MADV_SEQUENTIAL);
    
    /* Fill in context */
    ctx->ps_fd = fd;
    ctx->ps_base = mapped;
    ctx->ps_size = size;
    ctx->cuda_device = cuda_device;
    ctx->initialized = true;
    
    /* Initialize internal state */
    ictx->public = *ctx;
    ictx->alloc_offset = 0;
    
    printf("nearmem: initialized %zu MB VRAM at %p\n", 
           size >> 20, mapped);
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_init_auto(nearmem_ctx_t *ctx)
{
    char path[256];
    
    if (find_psdisk(path, sizeof(path)) != 0)
        return NEARMEM_ERROR_NO_DEVICE;
    
    return nearmem_init(ctx, path, 0);
}

void nearmem_shutdown(nearmem_ctx_t *ctx)
{
    if (!ctx || !ctx->initialized)
        return;
    
    /* Sync before shutdown */
    nearmem_sync(ctx, NEARMEM_SYNC_FULL);
    
    /* Unmap VRAM */
    if (ctx->ps_base && ctx->ps_size > 0)
        munmap(ctx->ps_base, ctx->ps_size);
    
    /* Close device */
    if (ctx->ps_fd >= 0)
        close(ctx->ps_fd);
    
    /* Destroy CUDA context */
    if (ctx->cuda_ctx)
        cuCtxDestroy(ctx->cuda_ctx);
    
    ctx->initialized = false;
}

size_t nearmem_get_capacity(nearmem_ctx_t *ctx)
{
    return ctx ? ctx->ps_size : 0;
}

/*
 * Memory region management
 *
 * The key insight: gpu_ptr is calculated as an offset within VRAM.
 * When we launch CUDA kernels, we pass this offset as the device pointer.
 * The GPU accesses the same physical memory that the CPU wrote to.
 */
nearmem_error_t nearmem_alloc(nearmem_ctx_t *ctx,
                               nearmem_region_t *region,
                               size_t size)
{
    nearmem_ctx_internal_t *ictx;
    size_t aligned_size;
    uint64_t offset;
    
    if (!ctx || !region || size == 0)
        return NEARMEM_ERROR_INVALID;
    
    /* Align to page size */
    aligned_size = (size + 4095) & ~4095UL;
    
    /* Simple bump allocator (for prototype) */
    ictx = get_internal_ctx(ctx);
    offset = ictx->alloc_offset;
    
    if (offset + aligned_size > ctx->ps_size)
        return NEARMEM_ERROR_BOUNDS;
    
    ictx->alloc_offset += aligned_size;
    
    /* Fill in region */
    region->cpu_ptr = (char*)ctx->ps_base + offset;
    region->gpu_ptr = (void*)(uintptr_t)offset;  /* Offset for CUDA */
    region->size = aligned_size;
    region->device_id = ctx->cuda_device;
    region->ps_fd = ctx->ps_fd;
    region->offset = offset;
    region->owned = true;
    
    return NEARMEM_OK;
}

void nearmem_free(nearmem_ctx_t *ctx, nearmem_region_t *region)
{
    /* Bump allocator doesn't support free - just leak for now */
    /* Production would use a proper allocator */
    (void)ctx;
    if (region) {
        region->cpu_ptr = NULL;
        region->gpu_ptr = NULL;
        region->size = 0;
    }
}

nearmem_error_t nearmem_map_offset(nearmem_ctx_t *ctx,
                                    nearmem_region_t *region,
                                    uint64_t offset,
                                    size_t size)
{
    if (!ctx || !region)
        return NEARMEM_ERROR_INVALID;
    
    if (offset + size > ctx->ps_size)
        return NEARMEM_ERROR_BOUNDS;
    
    region->cpu_ptr = (char*)ctx->ps_base + offset;
    region->gpu_ptr = (void*)(uintptr_t)offset;
    region->size = size;
    region->device_id = ctx->cuda_device;
    region->ps_fd = ctx->ps_fd;
    region->offset = offset;
    region->owned = false;
    
    return NEARMEM_OK;
}

/*
 * Synchronization
 *
 * This is critical. CPU and GPU have different memory models.
 *
 * CPU_TO_GPU:
 *   - CPU writes go through write-combining buffers
 *   - Must flush WC buffers before GPU can see writes
 *   - sfence isn't enough - need clflush or full barrier
 *
 * GPU_TO_CPU:
 *   - GPU writes may be in L2 or pending in memory controller
 *   - cuCtxSynchronize waits for all GPU work
 *   - CPU cache may hold stale data - need to invalidate
 */
nearmem_error_t nearmem_sync(nearmem_ctx_t *ctx, nearmem_sync_t type)
{
    if (!ctx || !ctx->initialized)
        return NEARMEM_ERROR_INVALID;
    
    switch (type) {
    case NEARMEM_SYNC_CPU_TO_GPU:
        /*
         * Flush CPU write-combine buffers.
         * On x86, this requires sfence + clflush on the region,
         * or a full memory barrier.
         */
        __sync_synchronize();  /* Full compiler + CPU barrier */
        
        /*
         * For write-combining memory, we also need to ensure the
         * WC buffers are flushed. msync with MS_SYNC forces this.
         */
        if (ctx->ps_base && ctx->ps_size > 0)
            msync(ctx->ps_base, ctx->ps_size, MS_SYNC);
        break;
        
    case NEARMEM_SYNC_GPU_TO_CPU:
        /*
         * Wait for all GPU operations to complete.
         */
        cuCtxSynchronize();
        
        /*
         * Invalidate CPU caches. For mmap'd device memory,
         * the kernel handles this when we access the pages.
         * But we can hint with madvise.
         */
        if (ctx->ps_base && ctx->ps_size > 0)
            madvise(ctx->ps_base, ctx->ps_size, MADV_DONTNEED);
        break;
        
    case NEARMEM_SYNC_FULL:
        nearmem_sync(ctx, NEARMEM_SYNC_CPU_TO_GPU);
        nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
        break;
    }
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_sync_region(nearmem_ctx_t *ctx,
                                     nearmem_region_t *region,
                                     nearmem_sync_t type)
{
    if (!ctx || !region)
        return NEARMEM_ERROR_INVALID;
    
    switch (type) {
    case NEARMEM_SYNC_CPU_TO_GPU:
        __sync_synchronize();
        msync(region->cpu_ptr, region->size, MS_SYNC);
        break;
        
    case NEARMEM_SYNC_GPU_TO_CPU:
        cuCtxSynchronize();
        madvise(region->cpu_ptr, region->size, MADV_DONTNEED);
        break;
        
    case NEARMEM_SYNC_FULL:
        nearmem_sync_region(ctx, region, NEARMEM_SYNC_CPU_TO_GPU);
        nearmem_sync_region(ctx, region, NEARMEM_SYNC_GPU_TO_CPU);
        break;
    }
    
    return NEARMEM_OK;
}

/*
 * Built-in operations
 *
 * These use CUDA driver API to launch kernels on the VRAM data.
 * The gpu_ptr is an offset within VRAM, which we convert to a
 * device pointer that CUDA can use.
 *
 * For the prototype, we use simple cuMemsetD8/cuMemcpyDtoD.
 * Production would have custom PTX kernels for each operation.
 */

/*
 * Convert region offset to CUDA device pointer
 *
 * This is the magic: the pseudoscopic device and CUDA both see
 * the same physical VRAM. The offset we calculated maps to the
 * same location whether accessed by CPU or GPU.
 */
static CUdeviceptr region_to_devptr(nearmem_region_t *region)
{
    /*
     * For direct BAR access, the GPU device pointer is just
     * the physical VRAM address. The offset within our mmap
     * corresponds directly to the offset within VRAM.
     *
     * Note: This assumes CUDA and pseudoscopic see the same GPU.
     * Multi-GPU systems need careful device matching.
     */
    return (CUdeviceptr)region->offset;
}

nearmem_error_t nearmem_memset(nearmem_ctx_t *ctx,
                                nearmem_region_t *region,
                                uint8_t value,
                                size_t offset,
                                size_t size)
{
    CUdeviceptr devptr;
    CUresult err;
    
    if (!ctx || !region)
        return NEARMEM_ERROR_INVALID;
    
    if (size == 0)
        size = region->size;
    
    if (offset + size > region->size)
        return NEARMEM_ERROR_BOUNDS;
    
    devptr = region_to_devptr(region) + offset;
    
    /*
     * cuMemsetD8 fills device memory with a byte value.
     * This runs at internal GPU bandwidth (~300 GB/s on modern GPUs).
     */
    err = cuMemsetD8(devptr, value, size);
    if (err != CUDA_SUCCESS)
        return NEARMEM_ERROR_CUDA;
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_memcpy(nearmem_ctx_t *ctx,
                                nearmem_region_t *dst_region,
                                size_t dst_offset,
                                nearmem_region_t *src_region,
                                size_t src_offset,
                                size_t size)
{
    CUdeviceptr dst_ptr, src_ptr;
    CUresult err;
    
    if (!ctx || !dst_region || !src_region)
        return NEARMEM_ERROR_INVALID;
    
    if (dst_offset + size > dst_region->size ||
        src_offset + size > src_region->size)
        return NEARMEM_ERROR_BOUNDS;
    
    dst_ptr = region_to_devptr(dst_region) + dst_offset;
    src_ptr = region_to_devptr(src_region) + src_offset;
    
    /*
     * cuMemcpyDtoD copies within device memory.
     * No PCIe traffic - all internal GPU bandwidth.
     */
    err = cuMemcpyDtoD(dst_ptr, src_ptr, size);
    if (err != CUDA_SUCCESS)
        return NEARMEM_ERROR_CUDA;
    
    return NEARMEM_OK;
}

/*
 * Stats
 */
void nearmem_get_stats(nearmem_ctx_t *ctx,
                       uint64_t *bytes_to_gpu,
                       uint64_t *bytes_from_gpu,
                       uint64_t *gpu_operations,
                       uint64_t *gpu_time_us)
{
    nearmem_ctx_internal_t *ictx = (nearmem_ctx_internal_t*)ctx;
    
    if (bytes_to_gpu)    *bytes_to_gpu = ictx->stats.bytes_to_gpu;
    if (bytes_from_gpu)  *bytes_from_gpu = ictx->stats.bytes_from_gpu;
    if (gpu_operations)  *gpu_operations = ictx->stats.gpu_operations;
    if (gpu_time_us)     *gpu_time_us = ictx->stats.gpu_time_us;
}

/*
 * Placeholder implementations for operations that need CUDA kernels
 * In production, these would be implemented with custom PTX code.
 */

nearmem_error_t nearmem_find(nearmem_ctx_t *ctx,
                              nearmem_region_t *region,
                              const void *pattern,
                              size_t pattern_len,
                              int64_t *result)
{
    /*
     * For prototype: fall back to CPU search
     * Production: launch GPU kernel for parallel search
     */
    const uint8_t *data = region->cpu_ptr;
    const uint8_t *pat = pattern;
    size_t i, j;
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    for (i = 0; i <= region->size - pattern_len; i++) {
        for (j = 0; j < pattern_len; j++) {
            if (data[i + j] != pat[j])
                break;
        }
        if (j == pattern_len) {
            *result = i;
            return NEARMEM_OK;
        }
    }
    
    *result = -1;
    return NEARMEM_OK;
}

nearmem_error_t nearmem_count_matches(nearmem_ctx_t *ctx,
                                       nearmem_region_t *region,
                                       const void *pattern,
                                       size_t pattern_len,
                                       uint64_t *count)
{
    const uint8_t *data = region->cpu_ptr;
    const uint8_t *pat = pattern;
    size_t i, j;
    uint64_t cnt = 0;
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    for (i = 0; i <= region->size - pattern_len; i++) {
        for (j = 0; j < pattern_len; j++) {
            if (data[i + j] != pat[j])
                break;
        }
        if (j == pattern_len)
            cnt++;
    }
    
    *count = cnt;
    return NEARMEM_OK;
}

nearmem_error_t nearmem_transform(nearmem_ctx_t *ctx,
                                   nearmem_region_t *region,
                                   const uint8_t lut[256])
{
    /*
     * For prototype: CPU transform
     * Production: GPU kernel applying LUT in parallel
     */
    uint8_t *data = region->cpu_ptr;
    size_t i;
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    for (i = 0; i < region->size; i++)
        data[i] = lut[data[i]];
    
    nearmem_sync(ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_histogram(nearmem_ctx_t *ctx,
                                   nearmem_region_t *region,
                                   uint64_t histogram[256])
{
    const uint8_t *data = region->cpu_ptr;
    size_t i;
    
    memset(histogram, 0, 256 * sizeof(uint64_t));
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    for (i = 0; i < region->size; i++)
        histogram[data[i]]++;
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_sort_u32(nearmem_ctx_t *ctx,
                                  nearmem_region_t *region,
                                  size_t count)
{
    /*
     * For prototype: qsort
     * Production: GPU radix sort
     */
    uint32_t *data = region->cpu_ptr;
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    int cmp(const void *a, const void *b) {
        return (*(uint32_t*)a > *(uint32_t*)b) - (*(uint32_t*)a < *(uint32_t*)b);
    }
    
    qsort(data, count, sizeof(uint32_t), cmp);
    
    nearmem_sync(ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_reduce_sum_f32(nearmem_ctx_t *ctx,
                                        nearmem_region_t *region,
                                        size_t count,
                                        float *result)
{
    const float *data = region->cpu_ptr;
    double sum = 0;
    size_t i;
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    for (i = 0; i < count; i++)
        sum += data[i];
    
    *result = (float)sum;
    return NEARMEM_OK;
}
