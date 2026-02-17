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
            if (strlen(entry->d_name) + 6 > path_len)
                continue;  /* Name too long, skip */
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
    
    char auto_path[256];

    if (!ctx)
        return NEARMEM_ERROR_INVALID;
    
    /* Auto-detect device if path is NULL */
    if (!device_path) {
        if (find_psdisk(auto_path, sizeof(auto_path)) == 0) {
            device_path = auto_path;
        } else {
            return NEARMEM_ERROR_NO_DEVICE;
        }
    }
    
    memset(ctx, 0, sizeof(*ctx));
    
#ifndef NEARMEM_USE_CUDA_HEADERS
    /* Load CUDA library */
    if (load_cuda() != 0) {
        return NEARMEM_ERROR_CUDA;
    }
#endif
    
    /* Initialize CUDA */
    cu_err = cuInit(0);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nearmem: cuInit failed: %d\n", cu_err);
        return NEARMEM_ERROR_CUDA;
    }
    
    /* Get CUDA device */
    cu_err = cuDeviceGet(&cu_dev, cuda_device);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nearmem: cuDeviceGet(%d) failed: %d\n", cuda_device, cu_err);
        return NEARMEM_ERROR_CUDA;
    }
    
    /* Create CUDA context */
    cu_err = cuCtxCreate((CUcontext*)&ctx->cuda_ctx, 0, cu_dev);
    if (cu_err != CUDA_SUCCESS) {
        fprintf(stderr, "nearmem: cuCtxCreate failed: %d\n", cu_err);
        return NEARMEM_ERROR_CUDA;
    }
    
    /* Open pseudoscopic block device */
    fd = open(device_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "nearmem: failed to open %s: %s\n", 
                device_path, strerror(errno));
        cuCtxDestroy(ctx->cuda_ctx);
        return NEARMEM_ERROR_NO_DEVICE;
    }
    
    /* Get device size */
    size = get_device_size(fd);
    if (size == 0) {
        fprintf(stderr, "nearmem: failed to get device size\n");
        close(fd);
        cuCtxDestroy(ctx->cuda_ctx);
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
        return NEARMEM_ERROR_MMAP;
    }
    
    /* Advise kernel about access pattern */
    madvise(mapped, size, MADV_SEQUENTIAL);
    
    /* Fill in public context */
    ctx->ps_fd = fd;
    ctx->ps_base = mapped;
    ctx->ps_size = size;
    ctx->cuda_device = cuda_device;
    ctx->initialized = true;
    
    /*
     * Allocate internal context and link it.
     * The ictx extends the public ctx; we store a snapshot of the public
     * portion as the first member so get_internal_ctx() can cast safely.
     * NOTE: The bump allocator state lives here, not in the public ctx.
     */
    ictx = calloc(1, sizeof(*ictx));
    if (!ictx) {
        fprintf(stderr, "nearmem: failed to allocate internal context\n");
        munmap(mapped, size);
        close(fd);
        cuCtxDestroy(ctx->cuda_ctx);
        return NEARMEM_ERROR_INIT;
    }
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

/*
 * Find PCI device with pseudoscopic driver bound
 * Returns the PCI address (e.g., "0000:06:00.0") in buf
 */
static int find_pseudoscopic_pci(char *buf, size_t buf_len)
{
    DIR *dir;
    struct dirent *entry;
    
    dir = opendir("/sys/bus/pci/drivers/pseudoscopic");
    if (!dir)
        return -1;
    
    while ((entry = readdir(dir)) != NULL) {
        /* Look for PCI addresses like 0000:06:00.0 */
        if (entry->d_name[0] == '0' && strchr(entry->d_name, ':')) {
            size_t name_len = strlen(entry->d_name);
            if (name_len >= buf_len)
                continue;  /* Name too long, skip */
            memcpy(buf, entry->d_name, name_len + 1);
            closedir(dir);
            return 0;
        }
    }
    
    closedir(dir);
    return -1;
}

/*
 * nearmem_init_bar1 - Initialize using direct BAR1 mmap
 *
 * This is a fallback when the pseudoscopic block device (/dev/psdisk*)
 * isn't available. It directly mmaps the PCI BAR1 resource from sysfs.
 *
 * Useful for:
 *   - GT 1030 and other GPUs where block device creation fails
 *   - Testing without full driver setup
 *   - Debugging driver issues
 *
 * Note: Requires root privileges to access sysfs resource files.
 */
nearmem_error_t nearmem_init_bar1(nearmem_ctx_t *ctx, 
                                   const char *pci_addr,
                                   int cuda_device)
{
    char pci_buf[256];
    char bar1_path[256];
    struct stat st;
    int fd;
    void *mapped;
    size_t size;
    CUdevice cu_dev;
    CUresult cu_err;
    
    if (!ctx)
        return NEARMEM_ERROR_INVALID;
    
    /* Auto-detect PCI address if not provided */
    if (!pci_addr || !pci_addr[0]) {
        if (find_pseudoscopic_pci(pci_buf, sizeof(pci_buf)) != 0) {
            fprintf(stderr, "nearmem: no pseudoscopic device found\n");
            return NEARMEM_ERROR_NO_DEVICE;
        }
        pci_addr = pci_buf;
    }
    
    /* Build path to BAR1 resource (use resource1_wc for write-combining) */
    snprintf(bar1_path, sizeof(bar1_path),
             "/sys/bus/pci/devices/%s/resource1_wc", pci_addr);
    
    /* Fall back to non-WC if WC not available */
    if (stat(bar1_path, &st) != 0) {
        snprintf(bar1_path, sizeof(bar1_path),
                 "/sys/bus/pci/devices/%s/resource1", pci_addr);
    }
    
    printf("nearmem: trying direct BAR1 access: %s\n", bar1_path);
    
    /* Get BAR1 size */
    if (stat(bar1_path, &st) != 0) {
        fprintf(stderr, "nearmem: cannot stat %s: %s\n", 
                bar1_path, strerror(errno));
        return NEARMEM_ERROR_NO_DEVICE;
    }
    size = st.st_size;
    
    if (size == 0) {
        fprintf(stderr, "nearmem: BAR1 size is 0\n");
        return NEARMEM_ERROR_NO_DEVICE;
    }
    
    /* Open BAR1 resource file */
    fd = open(bar1_path, O_RDWR | O_SYNC);
    if (fd < 0) {
        fprintf(stderr, "nearmem: cannot open %s: %s\n",
                bar1_path, strerror(errno));
        fprintf(stderr, "nearmem: hint: run as root or add user to 'video' group\n");
        return NEARMEM_ERROR_NO_DEVICE;
    }
    
    /* mmap the BAR1 */
    mapped = mmap(NULL, size, PROT_READ | PROT_WRITE,
                  MAP_SHARED, fd, 0);
    if (mapped == MAP_FAILED) {
        fprintf(stderr, "nearmem: mmap BAR1 failed: %s\n", strerror(errno));
        close(fd);
        return NEARMEM_ERROR_MMAP;
    }
    
    /* Fill in context first */
    memset(ctx, 0, sizeof(*ctx));
    ctx->ps_fd = fd;
    ctx->ps_base = mapped;
    ctx->ps_size = size;
    ctx->cuda_device = cuda_device;
    ctx->initialized = true;
    
#ifndef NEARMEM_USE_CUDA_HEADERS
    /* Load CUDA library (optional - CPU mode works without it) */
    if (load_cuda() != 0) {
        fprintf(stderr, "nearmem: CUDA not available (CPU-only mode)\n");
        /* Continue without CUDA */
    } else
#endif
    {
        /* Initialize CUDA if available */
        cu_err = cuInit(0);
        if (cu_err == CUDA_SUCCESS) {
            cu_err = cuDeviceGet(&cu_dev, cuda_device);
            if (cu_err == CUDA_SUCCESS) {
                cu_err = cuCtxCreate((CUcontext*)&ctx->cuda_ctx, 0, cu_dev);
                if (cu_err != CUDA_SUCCESS) {
                    ctx->cuda_ctx = NULL;
                    fprintf(stderr, "nearmem: CUDA context creation failed\n");
                }
            }
        }
    }
    
    printf("nearmem: BAR1 direct access initialized\n");
    printf("nearmem: PCI: %s, Size: %zu MB, Base: %p\n",
           pci_addr, size >> 20, mapped);
    
    return NEARMEM_OK;
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
        if (cuCtxSynchronize)
            cuCtxSynchronize();
        
        /*
         * CPU cache invalidation for device-mapped memory.
         *
         * CRITICAL: Do NOT use MADV_DONTNEED here! On Linux, that
         * tells the kernel to DISCARD the pages, destroying any data
         * the GPU wrote. For device-mapped (BAR1) pages, the mapping
         * is typically uncacheable/write-combining anyway, so CPU
         * cache stales are not a concern. A full memory barrier is
         * sufficient to order subsequent reads after the GPU sync.
         *
         * If the mapping uses write-back caching (rare for BAR1),
         * we would need per-cacheline clflush. But for the common
         * MAP_SHARED + O_SYNC path, __sync_synchronize is enough.
         */
        __sync_synchronize();
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
        if (region->cpu_ptr && region->size > 0)
            msync(region->cpu_ptr, region->size, MS_SYNC);
        break;
        
    case NEARMEM_SYNC_GPU_TO_CPU:
        if (cuCtxSynchronize)
            cuCtxSynchronize();
        /* See nearmem_sync() comment: no MADV_DONTNEED on device memory */
        __sync_synchronize();
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
     * CPU fallback with prefetch hints.
     * For single-byte patterns, use memchr for libc-optimized SIMD.
     * Production: launch GPU kernel for parallel search.
     */
    const uint8_t *data = region->cpu_ptr;
    const uint8_t *pat = pattern;
    
    if (!data || !pat || pattern_len == 0 || pattern_len > region->size) {
        *result = -1;
        return NEARMEM_OK;
    }
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    /* Fast path: single-byte search via libc memchr (SSE/AVX optimized) */
    if (pattern_len == 1) {
        const void *found = memchr(data, pat[0], region->size);
        *result = found ? (const uint8_t*)found - data : -1;
        return NEARMEM_OK;
    }
    
    /* Multi-byte: scan with first-byte filter + software prefetch */
    size_t end = region->size - pattern_len;
    uint8_t first = pat[0];
    
    for (size_t i = 0; i <= end; i++) {
        /* Prefetch 2 cache lines ahead (~128 bytes) */
        if (__builtin_expect(i + 128 <= end, 1))
            __builtin_prefetch(data + i + 128, 0, 0);
            
        /* First-byte filter: skip quickly if first byte doesn't match */
        if (data[i] != first)
            continue;
        
        /* Full pattern comparison (skip first byte, already matched) */
        if (memcmp(data + i + 1, pat + 1, pattern_len - 1) == 0) {
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
    uint64_t cnt = 0;
    
    if (!data || !pat || pattern_len == 0 || pattern_len > region->size) {
        *count = 0;
        return NEARMEM_OK;
    }
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    size_t end = region->size - pattern_len;
    uint8_t first = pat[0];
    
    for (size_t i = 0; i <= end; i++) {
        if (__builtin_expect(i + 128 <= end, 1))
            __builtin_prefetch(data + i + 128, 0, 0);
        
        if (data[i] != first)
            continue;
            
        if (pattern_len == 1 || memcmp(data + i + 1, pat + 1, pattern_len - 1) == 0)
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

/* Portable comparator (the nested function was a GCC extension) */
static int cmp_u32(const void *a, const void *b)
{
    uint32_t va = *(const uint32_t*)a;
    uint32_t vb = *(const uint32_t*)b;
    return (va > vb) - (va < vb);
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
    
    qsort(data, count, sizeof(uint32_t), cmp_u32);
    
    nearmem_sync(ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    return NEARMEM_OK;
}

nearmem_error_t nearmem_reduce_sum_f32(nearmem_ctx_t *ctx,
                                        nearmem_region_t *region,
                                        size_t count,
                                        float *result)
{
    const float *data = region->cpu_ptr;
    size_t i;
    
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    /*
     * Kahan compensated summation for better numerical accuracy.
     * Important when accumulating many small floats over VRAM-sized buffers.
     */
    double sum = 0.0;
    double comp = 0.0;  /* Running compensation for lost low-order bits */
    
    for (i = 0; i < count; i++) {
        /* Prefetch: float is 4 bytes, cache line is 64, so 16 elements/line */
        if (__builtin_expect(i + 64 < count, 1))
            __builtin_prefetch(data + i + 64, 0, 1);
        
        double y = (double)data[i] - comp;
        double t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    
    *result = (float)sum;
    return NEARMEM_OK;
}
