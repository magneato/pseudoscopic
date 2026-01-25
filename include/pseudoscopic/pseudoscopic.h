/* SPDX-License-Identifier: GPL-2.0 */
/*
 * pseudoscopic.h - The Abyssal Chart
 *
 * Reversing depth perception in the memory hierarchy.
 *
 * This header maps the internal structures of the driver, defining
 * the vessel (ps_device) that traverses the memory hierarchy.
 *
 * Theme: Bioluminescence - illuminating the dark paths of data.
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
#include <linux/cdev.h>
#include <linux/memremap.h>
#include <linux/migrate.h>
#include <linux/mmu_notifier.h>
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/blk-mq.h>
#include <linux/blkdev.h>
#include <linux/version.h>
/* gendisk.h merged into blkdev.h in kernel 5.18+ */
#if LINUX_VERSION_CODE < KERNEL_VERSION(5, 18, 0)
#include <linux/gendisk.h>
#endif

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
 * Logging Macros - The Bioluminescent Beacons
 * Using standard kernel prefixes styled for our domain.
 */
#define ps_fmt(fmt) KBUILD_MODNAME ": " fmt

#define ps_err(ps_dev, fmt, ...) \
    dev_err(&(ps_dev)->pdev->dev, ps_fmt(fmt), ##__VA_ARGS__)

#define ps_warn(ps_dev, fmt, ...) \
    dev_warn(&(ps_dev)->pdev->dev, ps_fmt(fmt), ##__VA_ARGS__)

#define ps_info(ps_dev, fmt, ...) \
    dev_info(&(ps_dev)->pdev->dev, ps_fmt(fmt), ##__VA_ARGS__)

/* Raw debug for early boot/probe logic where dev might be NULL */
#define ps_dbg_raw(fmt, ...) \
    pr_debug(ps_fmt(fmt), ##__VA_ARGS__)

#ifdef DEBUG
#define ps_dbg(ps_dev, fmt, ...) \
    dev_dbg(&(ps_dev)->pdev->dev, ps_fmt(fmt), ##__VA_ARGS__)
#else
#define ps_dbg(ps_dev, fmt, ...) \
    no_printk(KERN_DEBUG ps_fmt(fmt), ##__VA_ARGS__)
#endif

/*
 * Hardware Constants - The Cartography
 * -------------------------------------
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

/*
 * Memory Constants - The Geometry of the Deep
 * -------------------------------------------
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
 * Operating Modes - The Manifestations
 * ------------------------------------
 * Determines how the organism presents itself to the system.
 */
enum ps_op_mode {
    PS_MODE_RAM,        /* HMM / ZONE_DEVICE - Extend system memory (Default) */
    PS_MODE_RAMDISK,    /* Block Device /dev/psdiskN - High-speed storage */
    PS_MODE_SWAP        /* Block Device /dev/pswapN - Swap-optimized storage */
};

/*
 * Statistics Counters - The Sonar Pings
 * -------------------------------------
 * Atomically updated, exposed via sysfs.
 */
enum ps_stat_type {
    PS_STAT_PAGE_FAULTS,            /* Total page faults handled */
    PS_STAT_MIGRATIONS_TO_RAM,      /* Pages migrated GPU → CPU */
    PS_STAT_MIGRATIONS_TO_DEV,      /* Pages migrated CPU → GPU */
    PS_STAT_ALLOC_FAIL,             /* Pool allocation failures */
    PS_STAT_DMA_FAIL,               /* DMA transfer failures */
    PS_STAT_NOTIFIER_INVALIDATE,    /* MMU notifier invalidations */
    PS_STAT_BLOCK_READ,             /* Block device reads */
    PS_STAT_BLOCK_WRITE,            /* Block device writes */
    PS_STAT_MAX
};

/*
 * Forward Declarations - The Shadows
 * ----------------------------------
 */
struct ps_device;
struct ps_pool;
struct ps_context;
struct ps_block_dev;

/*
 * struct ps_block_dev - The Devouring Maw
 * @disk:       Gendisk structure for block interface
 * @tag_set:    Block multi-queue tag set
 * @queue:      Request queue
 * @lock:       Protects block operations
 *
 * Used when operating in Ramdisk or Swap modes.
 * Presents VRAM as /dev/psdiskN or /dev/pswapN.
 */
struct ps_block_dev {
    struct gendisk          *disk;
    struct blk_mq_tag_set   tag_set;
    struct request_queue    *queue;
    spinlock_t              lock;
};

/*
 * struct ps_pool - The Abyssal Reservoir
 * @bitmap:     Allocation bitmap (1 = allocated)
 * @nr_pages:   Total pages in pool
 * @free_pages: Currently free pages
 * @lock:       Protects pool operations
 *
 * Bitmap-based allocator for O(1) allocation with
 * efficient memory usage tracking.
 */
struct ps_pool {
    unsigned long           *bitmap;
    unsigned long           nr_pages;
    unsigned long           free_pages;
    spinlock_t              lock;
};

/*
 * struct ps_device - The Vessel
 * @pdev:           Underlying PCI device
 * @cdev:           Character device (for ioctl interface)
 * @id:             Device index (0, 1, 2...)
 * @mode:           Operating mode (RAM, RAMDISK, SWAP)
 * @initialized:    Device fully initialized
 * @dying:          Device being removed
 * @vram:           Mapped VRAM region (write-combining)
 * @vram_base:      Physical address of VRAM BAR
 * @vram_size:      Size of mapped VRAM in bytes
 * @vram_res:       Pseudo-physical resource for HMM
 * @pagemap:        Device memory description for HMM
 * @pfn_first:      First PFN in our range
 * @pfn_last:       Last PFN in our range (exclusive)
 * @pool:           VRAM page pool for allocation
 * @bdev:           Block device (only in RAMDISK/SWAP modes)
 * @lock:           Protects device state changes
 * @stats:          Per-device statistics counters
 *
 * One instance per GPU. Created during PCI probe,
 * destroyed during PCI remove. The vessel that carries
 * data through the memory hierarchy.
 */
struct ps_device {
    struct pci_dev          *pdev;
    struct cdev             cdev;
    int                     id;
    
    /* Configuration */
    enum ps_op_mode         mode;
    
    /* State flags */
    bool                    initialized;
    bool                    dying;
    
    /* BAR1 mapping - The Airlock */
    void __iomem            *vram;
    resource_size_t         vram_base;
    resource_size_t         vram_size;
    
    /* HMM integration - The Deep Interface */
    struct resource         *vram_res;
    struct dev_pagemap      pagemap;
    unsigned long           pfn_first;
    unsigned long           pfn_last;
    
    /* Subsystems */
    struct ps_pool          *pool;      /* The Reservoir */
    struct ps_block_dev     *bdev;      /* Active in BLOCK/SWAP modes */
    
    /* Synchronization */
    struct mutex            lock;
    
    /* Telemetry - The Sonar */
    atomic64_t              stats[PS_STAT_MAX];
};

/*
 * struct ps_context - A Diver's Session
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
 * Core Module Functions - The Hatch
 * ---------------------------------
 * Defined in src/core/module.c
 */

/* Module initialization/cleanup */
int __init ps_module_init(void);
void __exit ps_module_exit(void);

/*
 * BAR Management Functions - The Airlock
 * --------------------------------------
 * Defined in src/core/bar.c
 */

/* Map/unmap GPU VRAM BAR */
int ps_bar_map(struct ps_device *dev);
void ps_bar_unmap(struct ps_device *dev);

/* Validate BAR configuration */
int ps_bar_validate(struct ps_device *dev);

/* Attempt BAR resize (for consumer GPUs) */
int ps_bar_resize(struct ps_device *dev, resource_size_t size);

/* Read from MMIO registers */
u32 ps_bar_read_mmio(struct ps_device *dev, u32 offset);

/*
 * Block Device Functions - The Devouring Maw
 * ------------------------------------------
 * Defined in src/core/block.c
 */

/* Initialize/cleanup block device interface */
int ps_block_init(struct ps_device *dev);
void ps_block_cleanup(struct ps_device *dev);

/*
 * Pool Management Functions - The Reservoir
 * -----------------------------------------
 * Defined in src/core/pool.c
 */

/* Create/destroy page pool */
int ps_pool_init(struct ps_device *dev);
void ps_pool_destroy(struct ps_device *dev);

/* Allocate/free pages from pool */
struct page *ps_pool_alloc(struct ps_pool *pool);
void ps_pool_free(struct ps_pool *pool, struct page *page);

/* Pool statistics */
unsigned long ps_pool_total(struct ps_pool *pool);
unsigned long ps_pool_free_count(struct ps_pool *pool);

/*
 * HMM Device Functions - The Deep Interface
 * -----------------------------------------
 * Defined in src/hmm/device.c
 */

/* Register/unregister device memory with kernel */
int ps_hmm_register(struct ps_device *dev);
void ps_hmm_unregister(struct ps_device *dev);

/* PFN validation and conversion */
bool ps_hmm_pfn_valid(struct ps_device *dev, unsigned long pfn);
unsigned long ps_hmm_page_to_offset(struct ps_device *dev, struct page *page);

/* Device pagemap operations */
extern const struct dev_pagemap_ops ps_devmem_ops;

/*
 * Migration Functions - The Currents
 * ----------------------------------
 * Defined in src/hmm/migrate.c
 */

/* Migrate pages between CPU and GPU */
int ps_migrate_to_device(struct ps_context *ctx, 
                         unsigned long start, 
                         unsigned long size);

vm_fault_t ps_migrate_to_ram(struct vm_fault *vmf);

/*
 * MMU Notifier Functions - The Tides
 * ----------------------------------
 * Defined in src/hmm/notifier.c
 */

/* Register/unregister notifier for address range */
int ps_notifier_register(struct ps_context *ctx,
                         unsigned long start,
                         unsigned long size);
void ps_notifier_unregister(struct ps_context *ctx);

/* Context management */
struct ps_context *ps_context_create(struct ps_device *dev,
                                     struct mm_struct *mm);
void ps_context_destroy(struct ps_context *ctx);

/* Notifier operations */
extern const struct mmu_interval_notifier_ops ps_notifier_ops;

/*
 * DMA Engine Functions - The Propulsion
 * -------------------------------------
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

int ps_dma_copy_sync(struct ps_device *dev,
                     void *dst,
                     const void *src,
                     size_t size,
                     bool to_vram);

/*
 * Assembly Functions - The Bioluminescent Speed
 * ----------------------------------------------
 * Defined in src/asm/ (memcpy_wc.asm, cache_ops.asm, barriers.asm)
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
 * Inline Helpers - The Quick Signals
 * ----------------------------------
 */

static inline struct ps_device *ps_page_to_dev(struct page *page)
{
    return page->pgmap ? page->pgmap->owner : NULL;
}

static inline void __iomem *ps_page_to_vram_addr(struct ps_device *dev,
                                                  struct page *page)
{
    unsigned long offset = ps_hmm_page_to_offset(dev, page);
    return dev->vram + offset;
}

static inline void ps_stats_inc(struct ps_device *dev, enum ps_stat_type stat)
{
    if (stat >= 0 && stat < PS_STAT_MAX)
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
 * Debug Helpers - The Echo Location
 * ---------------------------------
 * Compile out in release builds.
 */

#endif /* _PSEUDOSCOPIC_H_ */
