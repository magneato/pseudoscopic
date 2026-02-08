/*
 * nearmem_gpu.h - Multi-GPU Support and Memory Location API
 *
 * This header provides:
 *   - Multi-GPU enumeration and selection
 *   - Memory location verification (is this pointer in GPU RAM?)
 *   - GPU RAM base address retrieval
 *   - Memory region attributes
 *
 * Copyright (C) 2025 Neural Splines LLC
 * License: MIT
 */

#ifndef _NEARMEM_GPU_H_
#define _NEARMEM_GPU_H_

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ════════════════════════════════════════════════════════════════════════════
 * GPU DEVICE INFORMATION
 * ════════════════════════════════════════════════════════════════════════════
 */

/* Maximum number of supported GPUs */
#define NEARMEM_MAX_GPUS    16

/* GPU device information */
typedef struct {
    int             index;              /* GPU index (0-based) */
    char            name[256];          /* Device name */
    char            pci_address[16];    /* PCI bus address (e.g., "0000:01:00.0") */
    uint16_t        vendor_id;          /* PCI vendor ID (0x10DE for NVIDIA) */
    uint16_t        device_id;          /* PCI device ID */
    
    /* Memory information */
    uint64_t        vram_size;          /* Total VRAM size in bytes */
    uint64_t        vram_available;     /* Available VRAM (not used by display) */
    uint64_t        bar1_base;          /* BAR1 physical base address */
    uint64_t        bar1_size;          /* BAR1 aperture size */
    
    /* Pseudoscopic device */
    char            block_device[32];   /* e.g., "/dev/psdisk0" */
    bool            ps_available;       /* Pseudoscopic device exists */
    int             ps_fd;              /* File descriptor (if opened) */
    
    /* Capabilities */
    bool            supports_hmm;       /* Heterogeneous Memory Management */
    bool            supports_p2p;       /* Peer-to-peer access */
    bool            supports_uvm;       /* Unified Virtual Memory */
    int             compute_capability; /* e.g., 70 for SM 7.0 */
    
    /* Current state */
    bool            is_active;          /* Currently selected for operations */
    uint64_t        allocated_bytes;    /* Bytes currently allocated */
} nearmem_gpu_info_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * GPU ENUMERATION
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * nearmem_gpu_count - Get number of available GPUs
 *
 * Returns: Number of GPUs found, or -1 on error
 */
int nearmem_gpu_count(void);

/*
 * nearmem_gpu_enumerate - Enumerate all available GPUs
 * @infos:      Array to fill with GPU information
 * @max_gpus:   Maximum number of GPUs to enumerate
 *
 * Returns: Number of GPUs enumerated, or -1 on error
 */
int nearmem_gpu_enumerate(nearmem_gpu_info_t *infos, int max_gpus);

/*
 * nearmem_gpu_get_info - Get information about a specific GPU
 * @index:      GPU index (0-based)
 * @info:       Output structure to fill
 *
 * Returns: 0 on success, -1 on error
 */
int nearmem_gpu_get_info(int index, nearmem_gpu_info_t *info);

/*
 * nearmem_gpu_select - Select GPU for subsequent operations
 * @index:      GPU index to select
 *
 * Returns: 0 on success, -1 on error
 */
int nearmem_gpu_select(int index);

/*
 * nearmem_gpu_current - Get currently selected GPU index
 *
 * Returns: Current GPU index, or -1 if none selected
 */
int nearmem_gpu_current(void);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MEMORY LOCATION API
 * ════════════════════════════════════════════════════════════════════════════
 */

/* Memory location types */
typedef enum {
    MEMLOC_UNKNOWN = 0,     /* Cannot determine location */
    MEMLOC_CPU,             /* System RAM */
    MEMLOC_GPU_VRAM,        /* GPU VRAM (via pseudoscopic) */
    MEMLOC_GPU_MANAGED,     /* CUDA managed memory */
    MEMLOC_PINNED,          /* Pinned CPU memory (DMA accessible) */
    MEMLOC_MAPPED,          /* Memory-mapped file or device */
} nearmem_memloc_t;

/* Memory location information */
typedef struct {
    nearmem_memloc_t    type;           /* Memory location type */
    int                 gpu_index;      /* GPU index (if GPU memory) */
    uint64_t            physical_addr;  /* Physical address (if known) */
    uint64_t            gpu_offset;     /* Offset within GPU VRAM */
    size_t              size;           /* Size of the allocation */
    bool                is_coherent;    /* CPU-GPU coherent */
    bool                is_cached;      /* CPU cached */
} nearmem_memloc_info_t;

/*
 * nearmem_get_memloc - Determine where a memory pointer resides
 * @ptr:        Pointer to check
 * @info:       Output structure for detailed information (can be NULL)
 *
 * Returns: Memory location type, or MEMLOC_UNKNOWN on error
 */
nearmem_memloc_t nearmem_get_memloc(const void *ptr, nearmem_memloc_info_t *info);

/*
 * nearmem_is_gpu_memory - Check if pointer is in GPU memory
 * @ptr:        Pointer to check
 *
 * Returns: true if in GPU VRAM, false otherwise
 */
bool nearmem_is_gpu_memory(const void *ptr);

/*
 * nearmem_is_cpu_memory - Check if pointer is in CPU memory
 * @ptr:        Pointer to check
 *
 * Returns: true if in system RAM, false otherwise
 */
bool nearmem_is_cpu_memory(const void *ptr);

/*
 * nearmem_get_gpu_for_ptr - Get GPU index for a GPU memory pointer
 * @ptr:        Pointer to check
 *
 * Returns: GPU index, or -1 if not GPU memory
 */
int nearmem_get_gpu_for_ptr(const void *ptr);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * GPU RAM BASE ADDRESS
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * nearmem_gpu_get_vram_base - Get the base address of GPU VRAM
 * @gpu_index:  GPU index
 *
 * Returns: Physical base address, or 0 on error
 *
 * Note: This is the BAR1 base address. User-space pointers from mmap
 * will have a different virtual address.
 */
uint64_t nearmem_gpu_get_vram_base(int gpu_index);

/*
 * nearmem_gpu_get_vram_size - Get total VRAM size
 * @gpu_index:  GPU index
 *
 * Returns: VRAM size in bytes, or 0 on error
 */
uint64_t nearmem_gpu_get_vram_size(int gpu_index);

/*
 * nearmem_gpu_get_vram_free - Get available VRAM
 * @gpu_index:  GPU index
 *
 * Returns: Free VRAM in bytes, or 0 on error
 */
uint64_t nearmem_gpu_get_vram_free(int gpu_index);

/*
 * nearmem_ptr_to_gpu_offset - Convert user pointer to GPU VRAM offset
 * @ptr:        User-space pointer (from nearmem_alloc or mmap)
 *
 * Returns: Offset within GPU VRAM, or (uint64_t)-1 on error
 */
uint64_t nearmem_ptr_to_gpu_offset(const void *ptr);

/*
 * nearmem_gpu_offset_to_ptr - Convert GPU offset to user pointer
 * @ctx:        Near-memory context
 * @offset:     Offset within GPU VRAM
 *
 * Returns: User-space pointer, or NULL on error
 *
 * Note: The offset must be within an active allocation.
 */
void *nearmem_gpu_offset_to_ptr(struct nearmem_ctx *ctx, uint64_t offset);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MULTI-GPU MEMORY OPERATIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * nearmem_gpu_copy - Copy memory between GPUs
 * @dst_gpu:    Destination GPU index
 * @dst_offset: Destination offset within GPU VRAM
 * @src_gpu:    Source GPU index
 * @src_offset: Source offset within GPU VRAM
 * @size:       Number of bytes to copy
 *
 * Returns: 0 on success, -1 on error
 *
 * Note: If peer-to-peer is available, uses direct copy.
 *       Otherwise, stages through system RAM.
 */
int nearmem_gpu_copy(int dst_gpu, uint64_t dst_offset,
                     int src_gpu, uint64_t src_offset,
                     size_t size);

/*
 * nearmem_gpu_broadcast - Copy from one GPU to all others
 * @src_gpu:    Source GPU index
 * @src_offset: Source offset within GPU VRAM
 * @size:       Number of bytes to copy
 *
 * Returns: 0 on success, -1 on error
 */
int nearmem_gpu_broadcast(int src_gpu, uint64_t src_offset, size_t size);

#ifdef __cplusplus
}
#endif

#endif /* _NEARMEM_GPU_H_ */
