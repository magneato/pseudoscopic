/*
 * nearmem_gpu.c - Multi-GPU Support and Memory Location Implementation
 *
 * Copyright (C) 2025 Neural Splines LLC
 * License: MIT
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/mman.h>

#include "nearmem.h"
#include "nearmem_gpu.h"

/*
 * ════════════════════════════════════════════════════════════════════════════
 * INTERNAL STATE
 * ════════════════════════════════════════════════════════════════════════════
 */

static nearmem_gpu_info_t g_gpus[NEARMEM_MAX_GPUS];
static int g_gpu_count = -1;  /* -1 = not enumerated yet */
static int g_current_gpu = 0;

/* Track active mmaps for memory location detection */
typedef struct {
    void *start;
    void *end;
    int gpu_index;
    uint64_t gpu_offset;
} mmap_region_t;

static mmap_region_t g_mmap_regions[256];
static int g_mmap_count = 0;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SYSFS/PROCFS HELPERS
 * ════════════════════════════════════════════════════════════════════════════
 */

/* Read a sysfs attribute */
static int read_sysfs_string(const char *path, char *buf, size_t buflen) {
    int fd = open(path, O_RDONLY);
    if (fd < 0) return -1;
    
    ssize_t n = read(fd, buf, buflen - 1);
    close(fd);
    
    if (n < 0) return -1;
    buf[n] = '\0';
    
    /* Strip trailing newline */
    if (n > 0 && buf[n-1] == '\n') buf[n-1] = '\0';
    
    return 0;
}

static uint64_t read_sysfs_hex(const char *path) {
    char buf[32];
    if (read_sysfs_string(path, buf, sizeof(buf)) != 0)
        return 0;
    return strtoull(buf, NULL, 16);
}

/* Parse PCI resource file for BAR addresses */
static int parse_pci_resource(const char *pci_path, uint64_t *bar1_base, uint64_t *bar1_size) {
    char resource_path[512];
    snprintf(resource_path, sizeof(resource_path), "%s/resource", pci_path);
    
    FILE *f = fopen(resource_path, "r");
    if (!f) return -1;
    
    char line[256];
    int bar_idx = 0;
    
    while (fgets(line, sizeof(line), f) && bar_idx < 6) {
        uint64_t start, end, flags;
        if (sscanf(line, "%lx %lx %lx", &start, &end, &flags) == 3) {
            if (bar_idx == 1 && start != 0) {  /* BAR1 */
                *bar1_base = start;
                *bar1_size = end - start + 1;
            }
        }
        bar_idx++;
    }
    
    fclose(f);
    return 0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * GPU ENUMERATION
 * ════════════════════════════════════════════════════════════════════════════
 */

static void enumerate_gpus_internal(void) {
    if (g_gpu_count >= 0) return;  /* Already enumerated */
    
    g_gpu_count = 0;
    memset(g_gpus, 0, sizeof(g_gpus));
    
    /* Scan PCI devices for NVIDIA GPUs */
    const char *pci_base = "/sys/bus/pci/devices";
    DIR *dir = opendir(pci_base);
    if (!dir) return;
    
    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL && g_gpu_count < NEARMEM_MAX_GPUS) {
        if (entry->d_name[0] == '.') continue;
        
        char path[512];
        snprintf(path, sizeof(path), "%s/%s/vendor", pci_base, entry->d_name);
        
        uint64_t vendor = read_sysfs_hex(path);
        if (vendor != 0x10DE) continue;  /* Not NVIDIA */
        
        /* Check device class (VGA or 3D controller) */
        snprintf(path, sizeof(path), "%s/%s/class", pci_base, entry->d_name);
        uint64_t class = read_sysfs_hex(path);
        uint32_t class_code = (class >> 8) & 0xFFFF;
        if (class_code != 0x0300 && class_code != 0x0302) continue;  /* Not display */
        
        /* Found an NVIDIA GPU */
        nearmem_gpu_info_t *gpu = &g_gpus[g_gpu_count];
        gpu->index = g_gpu_count;
        gpu->vendor_id = 0x10DE;
        
        strncpy(gpu->pci_address, entry->d_name, sizeof(gpu->pci_address) - 1);
        
        /* Read device ID */
        snprintf(path, sizeof(path), "%s/%s/device", pci_base, entry->d_name);
        gpu->device_id = read_sysfs_hex(path);
        
        /* Try to get device name from nvidia-smi or use device ID */
        snprintf(gpu->name, sizeof(gpu->name), "NVIDIA GPU %04X", gpu->device_id);
        
        /* Parse BAR1 info */
        snprintf(path, sizeof(path), "%s/%s", pci_base, entry->d_name);
        parse_pci_resource(path, &gpu->bar1_base, &gpu->bar1_size);
        
        /* Estimate VRAM from BAR1 size (often equal for desktop GPUs) */
        gpu->vram_size = gpu->bar1_size;
        gpu->vram_available = gpu->vram_size;
        
        /* Check for pseudoscopic device */
        snprintf(gpu->block_device, sizeof(gpu->block_device), 
                 "/dev/psdisk%d", g_gpu_count);
        
        struct stat st;
        gpu->ps_available = (stat(gpu->block_device, &st) == 0);
        
        /* Set capabilities (conservative defaults) */
        gpu->supports_hmm = false;
        gpu->supports_p2p = false;
        gpu->supports_uvm = false;
        gpu->compute_capability = 0;
        
        g_gpu_count++;
    }
    
    closedir(dir);
}

int nearmem_gpu_count(void) {
    enumerate_gpus_internal();
    return g_gpu_count;
}

int nearmem_gpu_enumerate(nearmem_gpu_info_t *infos, int max_gpus) {
    enumerate_gpus_internal();
    
    int count = (g_gpu_count < max_gpus) ? g_gpu_count : max_gpus;
    memcpy(infos, g_gpus, count * sizeof(nearmem_gpu_info_t));
    
    return count;
}

int nearmem_gpu_get_info(int index, nearmem_gpu_info_t *info) {
    enumerate_gpus_internal();
    
    if (index < 0 || index >= g_gpu_count || !info)
        return -1;
    
    *info = g_gpus[index];
    return 0;
}

int nearmem_gpu_select(int index) {
    enumerate_gpus_internal();
    
    if (index < 0 || index >= g_gpu_count)
        return -1;
    
    g_current_gpu = index;
    return 0;
}

int nearmem_gpu_current(void) {
    return g_current_gpu;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MEMORY LOCATION API
 * ════════════════════════════════════════════════════════════════════════════
 */

/* Register an mmap region for tracking */
void nearmem_register_mmap(void *start, size_t size, int gpu_index, uint64_t offset) {
    if (g_mmap_count >= 256) return;
    
    g_mmap_regions[g_mmap_count].start = start;
    g_mmap_regions[g_mmap_count].end = (char*)start + size;
    g_mmap_regions[g_mmap_count].gpu_index = gpu_index;
    g_mmap_regions[g_mmap_count].gpu_offset = offset;
    g_mmap_count++;
}

/* Unregister an mmap region */
void nearmem_unregister_mmap(void *start) {
    for (int i = 0; i < g_mmap_count; i++) {
        if (g_mmap_regions[i].start == start) {
            /* Shift remaining entries */
            for (int j = i; j < g_mmap_count - 1; j++) {
                g_mmap_regions[j] = g_mmap_regions[j + 1];
            }
            g_mmap_count--;
            return;
        }
    }
}

nearmem_memloc_t nearmem_get_memloc(const void *ptr, nearmem_memloc_info_t *info) {
    if (!ptr) {
        if (info) {
            memset(info, 0, sizeof(*info));
            info->type = MEMLOC_UNKNOWN;
        }
        return MEMLOC_UNKNOWN;
    }
    
    /* Check registered mmap regions (GPU memory) */
    for (int i = 0; i < g_mmap_count; i++) {
        if (ptr >= g_mmap_regions[i].start && ptr < g_mmap_regions[i].end) {
            if (info) {
                info->type = MEMLOC_GPU_VRAM;
                info->gpu_index = g_mmap_regions[i].gpu_index;
                info->gpu_offset = g_mmap_regions[i].gpu_offset + 
                                  ((char*)ptr - (char*)g_mmap_regions[i].start);
                info->physical_addr = 0;  /* Not easily available */
                info->is_coherent = false;
                info->is_cached = false;
            }
            return MEMLOC_GPU_VRAM;
        }
    }
    
    /* Check /proc/self/maps to determine memory type */
    FILE *maps = fopen("/proc/self/maps", "r");
    if (!maps) {
        if (info) {
            memset(info, 0, sizeof(*info));
            info->type = MEMLOC_UNKNOWN;
        }
        return MEMLOC_UNKNOWN;
    }
    
    char line[512];
    uintptr_t addr = (uintptr_t)ptr;
    nearmem_memloc_t result = MEMLOC_UNKNOWN;
    
    while (fgets(line, sizeof(line), maps)) {
        uintptr_t start, end;
        char perms[8], path[256];
        path[0] = '\0';
        
        if (sscanf(line, "%lx-%lx %7s %*s %*s %*s %255s", 
                   &start, &end, perms, path) >= 3) {
            if (addr >= start && addr < end) {
                /* Found the region */
                if (path[0] == '\0' || strcmp(path, "[heap]") == 0 ||
                    strcmp(path, "[stack]") == 0) {
                    result = MEMLOC_CPU;
                } else if (strstr(path, "/dev/psdisk") != NULL) {
                    result = MEMLOC_GPU_VRAM;
                } else if (strstr(path, "/dev/") != NULL) {
                    result = MEMLOC_MAPPED;
                } else {
                    result = MEMLOC_CPU;  /* Regular mmap or library */
                }
                
                if (info) {
                    info->type = result;
                    info->size = end - start;
                    if (result == MEMLOC_GPU_VRAM) {
                        /* Try to determine GPU index from device path */
                        char *psdisk = strstr(path, "psdisk");
                        if (psdisk) {
                            info->gpu_index = atoi(psdisk + 6);
                        }
                    }
                }
                break;
            }
        }
    }
    
    fclose(maps);
    
    if (info && result == MEMLOC_UNKNOWN) {
        memset(info, 0, sizeof(*info));
        info->type = MEMLOC_UNKNOWN;
    }
    
    return result;
}

bool nearmem_is_gpu_memory(const void *ptr) {
    return nearmem_get_memloc(ptr, NULL) == MEMLOC_GPU_VRAM;
}

bool nearmem_is_cpu_memory(const void *ptr) {
    nearmem_memloc_t loc = nearmem_get_memloc(ptr, NULL);
    return loc == MEMLOC_CPU || loc == MEMLOC_PINNED;
}

int nearmem_get_gpu_for_ptr(const void *ptr) {
    nearmem_memloc_info_t info;
    nearmem_memloc_t loc = nearmem_get_memloc(ptr, &info);
    
    if (loc == MEMLOC_GPU_VRAM) {
        return info.gpu_index;
    }
    return -1;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * GPU RAM BASE ADDRESS
 * ════════════════════════════════════════════════════════════════════════════
 */

uint64_t nearmem_gpu_get_vram_base(int gpu_index) {
    enumerate_gpus_internal();
    
    if (gpu_index < 0 || gpu_index >= g_gpu_count)
        return 0;
    
    return g_gpus[gpu_index].bar1_base;
}

uint64_t nearmem_gpu_get_vram_size(int gpu_index) {
    enumerate_gpus_internal();
    
    if (gpu_index < 0 || gpu_index >= g_gpu_count)
        return 0;
    
    return g_gpus[gpu_index].vram_size;
}

uint64_t nearmem_gpu_get_vram_free(int gpu_index) {
    enumerate_gpus_internal();
    
    if (gpu_index < 0 || gpu_index >= g_gpu_count)
        return 0;
    
    /* TODO: Track allocations to compute free space */
    return g_gpus[gpu_index].vram_available;
}

uint64_t nearmem_ptr_to_gpu_offset(const void *ptr) {
    for (int i = 0; i < g_mmap_count; i++) {
        if (ptr >= g_mmap_regions[i].start && ptr < g_mmap_regions[i].end) {
            return g_mmap_regions[i].gpu_offset + 
                   ((char*)ptr - (char*)g_mmap_regions[i].start);
        }
    }
    return (uint64_t)-1;
}

void *nearmem_gpu_offset_to_ptr(nearmem_ctx_t *ctx, uint64_t offset) {
    if (!ctx || !ctx->ps_base)
        return NULL;
    
    if (offset >= ctx->ps_size)
        return NULL;
    
    return (char*)ctx->ps_base + offset;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MULTI-GPU MEMORY OPERATIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

int nearmem_gpu_copy(int dst_gpu, uint64_t dst_offset,
                     int src_gpu, uint64_t src_offset,
                     size_t size) {
    enumerate_gpus_internal();
    
    if (dst_gpu < 0 || dst_gpu >= g_gpu_count ||
        src_gpu < 0 || src_gpu >= g_gpu_count)
        return -1;
    
    /* For now, stage through CPU memory */
    /* TODO: Use P2P if available */
    
    void *staging = malloc(size);
    if (!staging) return -1;
    
    /* Open source device */
    int src_fd = open(g_gpus[src_gpu].block_device, O_RDONLY);
    if (src_fd < 0) {
        free(staging);
        return -1;
    }
    
    /* Read from source */
    lseek(src_fd, src_offset, SEEK_SET);
    ssize_t n = read(src_fd, staging, size);
    close(src_fd);
    
    if (n != (ssize_t)size) {
        free(staging);
        return -1;
    }
    
    /* Open destination device */
    int dst_fd = open(g_gpus[dst_gpu].block_device, O_WRONLY);
    if (dst_fd < 0) {
        free(staging);
        return -1;
    }
    
    /* Write to destination */
    lseek(dst_fd, dst_offset, SEEK_SET);
    n = write(dst_fd, staging, size);
    close(dst_fd);
    
    free(staging);
    
    return (n == (ssize_t)size) ? 0 : -1;
}

int nearmem_gpu_broadcast(int src_gpu, uint64_t src_offset, size_t size) {
    enumerate_gpus_internal();
    
    if (src_gpu < 0 || src_gpu >= g_gpu_count)
        return -1;
    
    int errors = 0;
    
    for (int i = 0; i < g_gpu_count; i++) {
        if (i == src_gpu) continue;
        
        if (nearmem_gpu_copy(i, src_offset, src_gpu, src_offset, size) != 0) {
            errors++;
        }
    }
    
    return errors ? -1 : 0;
}
