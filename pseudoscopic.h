/* SPDX-License-Identifier: GPL-2.0 */
/*
 * pseudoscopic.h - GPU VRAM as System RAM
 *
 * Reversing depth perception in the memory hierarchy.
 *
 * Copyright (C) 2025 Neural Splines LLC
 * Author: Robert L. Sitton, Jr.
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License version 2 as
 * published by the Free Software Foundation.
 */

#ifndef _PSEUDOSCOPIC_H_
#define _PSEUDOSCOPIC_H_

#include <linux/types.h>
#include <linux/pci.h>
#include <linux/memremap.h>
#include <linux/migrate.h>
#include <linux/mmu_notifier.h>

/*
 * Version: 0.0.x indicates pre-release development
 * 
 * Versioning follows: MAJOR.MINOR.PATCH
 *   MAJOR: Breaking changes to userspace interface
 *   MINOR: New features, backward compatible
 *   PATCH: Bug fixes only
 */
#define PS_VERSION_MAJOR    0
#define PS_VERSION_MINOR    0
#define PS_VERSION_PATCH    1
#define PS_VERSION_STRING   "0.0.1"

/*
 * Hardware Constants
 * ------------------
 * NVIDIA GPUs expose memory through three BARs:
 *   BAR0: MMIO registers (16-32 MB)
 *   BAR1: VRAM aperture (up to full VRAM on datacenter cards)
 *   BAR2/3: RAMIN (internal structures)
 *
 * We exclusively use BAR1 for VRAM access.
 */
#define PS_BAR_MMIO         0
#define PS_BAR_VRAM         1
#define PS_BAR_RAMIN        2

/* PCI vendor/device IDs for supported hardware */
#define PCI_VENDOR_ID_NVIDIA    0x10de

/* Tesla/Datacenter GPU device IDs with Large BAR support */
#define PS_DEV_TESLA_P100_PCIE      0x15f8  /* GP100GL */
#define PS_DEV_TESLA_P100_SXM2      0x15f9  /* GP100GL */
#define PS_DEV_TESLA_P40            0x1b38  /* GP102GL */
#define PS_DEV_TESLA_V100_PCIE      0x1db4  /* GV100GL */
#define PS_DEV_TESLA_V100_SXM2      0x1db1  /* GV100GL */
#define PS_DEV_TESLA_V100S          0x1df6  /* GV100GL */
#define PS_DEV_QUADRO_RTX_8000      0x1e30  /* TU102GL */
#define PS_DEV_QUADRO_RTX_6000      0x1e30  /* TU102GL */
#define PS_DEV_A100_PCIE            0x20f1  /* GA100 */
#define PS_DEV_A100_SXM4            0x20b0  /* GA100 */

/*
 * Memory Constants
 * ----------------
 * These define the geometry of our page management.
 */
#define PS_PAGE_SHIFT       12
#define PS_PAGE_SIZE        (1UL << PS_PAGE_SHIFT)
#define PS_PAGE_MASK        (~(PS_PAGE_SIZE - 1))

/* Hugepage support (2MB) */
#define PS_HUGEPAGE_SHIFT   21
#define PS_HUGEPAGE_SIZE    (1UL << PS_HUGEPAGE_SHIFT)
#define PS_HUGEPAGE_MASK    (~(PS_HUGEPAGE_SIZE - 1))

/* Maximum supported VRAM (for sanity checks) */
#define PS_MAX_VRAM_SIZE    (128ULL << 30)  /* 128 GB */

/*
 * Statistics Counters
 * -------------------
 * Atomically updated, exposed via sysfs.
 */
enum ps_stat_type {
    PS_STAT_MIGRATIONS_TO_RAM,      /* Pages migrated GPU → CPU */
    PS_STAT_MIGRATIONS_TO_DEV,      /* Pages migrated CPU → GPU */
    PS_STAT_PAGE_FAULTS,            /* Total page faults handled */
    PS_STAT_ALLOC_FAIL,             /* Pool allocation failures */
    PS_STAT_DMA_FAIL,               /* DMA transfer failures */
    PS_STAT_NOTIFIER_INVALIDATE,    /* MMU notifier invalidations */
    PS_STAT_MAX
};

/*
 * Forward Declarations
 * --------------------
 * Internal structures defined in source files.
 */
struct ps_device;
struct ps_pool;
struct ps_context;

/*
 * struct ps_device - Primary device structure
 * @pdev:       Underlying PCI device
 * @vram:       Mapped VRAM region (write-combining)
 * @vram_base:  Physical address of VRAM BAR
 * @vram_size:  Size of mapped VRAM in bytes
 * @pagemap:    Device memory description for HMM
 * @pool:       VRAM page pool for allocation
 * @stats:      Per-device statistics counters
 * @lock:       Protects device state changes
 *
 * One instance per GPU. Created during PCI probe,
 * destroyed during PCI remove.
 */
struct ps_device {
    struct pci_dev          *pdev;
    
    /* BAR1 mapping */
    void __iomem            *vram;
    resource_size_t         vram_base;
    resource_size_t         vram_size;
    
    /* HMM integration */
    struct dev_pagemap      pagemap;
    struct resource         *vram_res;
    unsigned long           pfn_first;
    unsigned long           pfn_last;
    
    /* Page management */
    struct ps_pool          *pool;
    
    /* Statistics */
    atomic64_t              stats[PS_STAT_MAX];
    
    /* Synchronization */
    spinlock_t              lock;
    
    /* State tracking */
    bool                    initialized;
    bool                    hmm_registered;
};

/*
 * struct ps_pool - VRAM page pool
 * @dev:        Parent device
 * @free_list:  Head of free page list
 * @total:      Total pages in pool
 * @free:       Currently free pages
 * @lock:       Protects pool operations
 *
 * Simple free-list allocator. Pages are pre-initialized
 * during driver load for fast allocation.
 */
struct ps_pool {
    struct ps_device        *dev;
    struct page             *free_list;
    unsigned long           total;
    atomic_long_t           free;
    spinlock_t              lock;
};

/*
 * struct ps_context - Per-process context
 * @dev:        Associated device
 * @mm:         Process memory map
 * @notifier:   MMU interval notifier
 * @node:       List node for context tracking
 *
 * Tracks per-process state for migration and
 * MMU notifier callbacks.
 */
struct ps_context {
    struct ps_device                *dev;
    struct mm_struct                *mm;
    struct mmu_interval_notifier    notifier;
    struct list_head                node;
};

/*
 * Core Module Functions
 * ---------------------
 * Defined in src/core/module.c
 */

/* Module initialization/cleanup */
int __init ps_module_init(void);
void __exit ps_module_exit(void);

/*
 * BAR Management Functions
 * ------------------------
 * Defined in src/core/bar.c
 */

/* Map/unmap GPU VRAM BAR */
int ps_bar_map(struct ps_device *dev);
void ps_bar_unmap(struct ps_device *dev);

/* Validate BAR configuration */
int ps_bar_validate(struct ps_device *dev);

/* Attempt BAR resize (for consumer GPUs) */
int ps_bar_resize(struct ps_device *dev, resource_size_t size);

/*
 * Pool Management Functions
 * -------------------------
 * Defined in src/core/pool.c
 */

/* Create/destroy page pool */
struct ps_pool *ps_pool_create(struct ps_device *dev);
void ps_pool_destroy(struct ps_pool *pool);

/* Allocate/free pages from pool */
struct page *ps_pool_alloc(struct ps_pool *pool);
void ps_pool_free(struct ps_pool *pool, struct page *page);

/* Pool statistics */
unsigned long ps_pool_total(struct ps_pool *pool);
unsigned long ps_pool_free_count(struct ps_pool *pool);

/*
 * HMM Device Functions
 * --------------------
 * Defined in src/hmm/device.c
 */

/* Register/unregister device memory with kernel */
int ps_hmm_register(struct ps_device *dev);
void ps_hmm_unregister(struct ps_device *dev);

/* Device pagemap operations */
extern const struct dev_pagemap_ops ps_devmem_ops;

/*
 * Migration Functions
 * -------------------
 * Defined in src/hmm/migrate.c
 */

/* Migrate pages between CPU and GPU */
int ps_migrate_to_device(struct ps_context *ctx, 
                         unsigned long start, 
                         unsigned long size);

vm_fault_t ps_migrate_to_ram(struct vm_fault *vmf);

/*
 * MMU Notifier Functions
 * ----------------------
 * Defined in src/hmm/notifier.c
 */

/* Register/unregister notifier for address range */
int ps_notifier_register(struct ps_context *ctx,
                         unsigned long start,
                         unsigned long size);
void ps_notifier_unregister(struct ps_context *ctx);

/* Notifier operations */
extern const struct mmu_interval_notifier_ops ps_notifier_ops;

/*
 * DMA Engine Functions
 * --------------------
 * Defined in src/dma/engine.c
 */

/* Initialize/shutdown DMA engine */
int ps_dma_init(struct ps_device *dev);
void ps_dma_shutdown(struct ps_device *dev);

/* Copy operations */
int ps_dma_copy_to_vram(struct ps_device *dev,
                        struct page *dst,
                        struct page *src);

int ps_dma_copy_from_vram(struct ps_device *dev,
                          struct page *dst,
                          struct page *src);

/*
 * Assembly Functions
 * ------------------
 * Defined in src/asm/*.asm
 *
 * These are the performance-critical hot paths,
 * hand-optimized for cache behavior and throughput.
 */

/* 
 * Memory copy with write-combining optimization.
 * Uses non-temporal stores to bypass cache.
 * 
 * @dst:   Destination (VRAM, WC-mapped)
 * @src:   Source (system RAM)
 * @count: Bytes to copy (must be multiple of 64)
 */
asmlinkage void ps_memcpy_to_vram(void *dst, const void *src, size_t count);

/*
 * Memory copy from VRAM to system RAM.
 * Uses prefetch hints for streaming reads.
 *
 * @dst:   Destination (system RAM)
 * @src:   Source (VRAM, WC-mapped)
 * @count: Bytes to copy (must be multiple of 64)
 */
asmlinkage void ps_memcpy_from_vram(void *dst, const void *src, size_t count);

/*
 * Flush cache lines to ensure coherency.
 * Uses clflushopt when available, clflush otherwise.
 *
 * @addr:  Start address (must be cache-aligned)
 * @count: Bytes to flush (must be multiple of 64)
 */
asmlinkage void ps_cache_flush(void *addr, size_t count);

/*
 * Write-back cache lines without invalidation.
 * Uses clwb when available, falls back to clflushopt.
 *
 * @addr:  Start address (must be cache-aligned)
 * @count: Bytes to write back (must be multiple of 64)
 */
asmlinkage void ps_cache_writeback(void *addr, size_t count);

/*
 * Memory barriers for ordering.
 */
asmlinkage void ps_sfence(void);  /* Store fence */
asmlinkage void ps_lfence(void);  /* Load fence */
asmlinkage void ps_mfence(void);  /* Full fence */

/*
 * Statistics Helpers
 * ------------------
 * Inline for minimal overhead.
 */

static inline void ps_stats_inc(struct ps_device *dev, enum ps_stat_type stat)
{
    atomic64_inc(&dev->stats[stat]);
}

static inline u64 ps_stats_read(struct ps_device *dev, enum ps_stat_type stat)
{
    return atomic64_read(&dev->stats[stat]);
}

static inline void ps_stats_reset(struct ps_device *dev, enum ps_stat_type stat)
{
    atomic64_set(&dev->stats[stat], 0);
}

/*
 * Page Helpers
 * ------------
 * Map between page structs and VRAM addresses.
 */

static inline struct ps_device *ps_page_to_dev(struct page *page)
{
    return page->zone_device_data;
}

static inline unsigned long ps_page_to_vram_offset(struct ps_device *dev,
                                                    struct page *page)
{
    unsigned long pfn = page_to_pfn(page);
    return (pfn - dev->pfn_first) << PAGE_SHIFT;
}

static inline void __iomem *ps_page_to_vram_addr(struct ps_device *dev,
                                                  struct page *page)
{
    return dev->vram + ps_page_to_vram_offset(dev, page);
}

/*
 * Debug Helpers
 * -------------
 * Compile out in release builds.
 */

#ifdef DEBUG
#define ps_dbg(dev, fmt, ...) \
    dev_dbg(&(dev)->pdev->dev, "pseudoscopic: " fmt, ##__VA_ARGS__)
#else
#define ps_dbg(dev, fmt, ...) do { } while (0)
#endif

#define ps_info(dev, fmt, ...) \
    dev_info(&(dev)->pdev->dev, "pseudoscopic: " fmt, ##__VA_ARGS__)

#define ps_warn(dev, fmt, ...) \
    dev_warn(&(dev)->pdev->dev, "pseudoscopic: " fmt, ##__VA_ARGS__)

#define ps_err(dev, fmt, ...) \
    dev_err(&(dev)->pdev->dev, "pseudoscopic: " fmt, ##__VA_ARGS__)

#endif /* _PSEUDOSCOPIC_H_ */
