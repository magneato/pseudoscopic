// SPDX-License-Identifier: GPL-2.0
/*
 * migrate.c - Page migration between CPU and GPU
 *
 * Implements transparent page migration using the kernel's
 * migrate_vma_*() APIs. When CPU touches a VRAM-resident page,
 * migrate_to_ram() is invoked to transparently copy it to
 * system RAM.
 *
 * The beauty: userspace code doesn't need to know pages are
 * in VRAM. The kernel's page fault handling triggers migration
 * automatically.
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#include <linux/migrate.h>
#include <linux/mm.h>
#include <linux/highmem.h>
#include <linux/pagemap.h>
#include <linux/dma-mapping.h>

#include <pseudoscopic/pseudoscopic.h>
#include <pseudoscopic/asm.h>

/*
 * ps_page_free - Device pagemap page_free callback
 * @page: Page being freed
 *
 * Called when a VRAM page's refcount drops to zero.
 * We return it to our free pool.
 */
static void ps_page_free(struct page *page)
{
    struct ps_device *dev = ps_page_to_dev(page);
    
    if (dev && dev->pool)
        ps_pool_free(dev->pool, page);
}

/*
 * ps_migrate_to_ram - Migrate page from VRAM to system RAM
 * @vmf: VM fault information
 *
 * This is the core callback invoked by the kernel when CPU
 * accesses a VRAM-resident page. We:
 *   1. Allocate a system RAM page
 *   2. Copy data from VRAM (via DMA or PIO)
 *   3. Use migrate_vma_*() to update page tables
 *   4. Return VRAM page to pool
 *
 * The migration is transparent to userspace - the faulting
 * instruction retries and succeeds.
 *
 * Returns: VM_FAULT_* code
 */
vm_fault_t ps_migrate_to_ram(struct vm_fault *vmf)
{
    struct page *device_page = vmf->page;
    struct ps_device *dev;
    struct page *sys_page = NULL;
    unsigned long src_pfns, dst_pfns;
    struct migrate_vma args = { 0 };
    void __iomem *vram_addr;
    void *sys_addr;
    int ret;
    
    /* Get our device from the page */
    dev = ps_page_to_dev(device_page);
    if (!dev || !dev->initialized) {
        pr_err("pseudoscopic: migrate_to_ram on invalid device\n");
        return VM_FAULT_SIGBUS;
    }
    
    ps_stats_inc(dev, PS_STAT_PAGE_FAULTS);
    ps_stats_inc(dev, PS_STAT_MIGRATIONS_TO_RAM);
    
    ps_dbg(dev, "migrate_to_ram: addr=0x%lx page=%px\n",
           vmf->address, device_page);
    
    /*
     * Allocate destination page in system RAM.
     *
     * GFP_HIGHUSER_MOVABLE: Can be anywhere in RAM, can be
     * migrated by compaction. Optimal for general use.
     */
    sys_page = alloc_page(GFP_HIGHUSER_MOVABLE);
    if (!sys_page) {
        ps_warn(dev, "migrate_to_ram: failed to allocate system page\n");
        return VM_FAULT_OOM;
    }
    
    /* Lock the system page for migration */
    lock_page(sys_page);
    
    /*
     * Set up migration arguments.
     *
     * MIGRATE_VMA_SELECT_DEVICE_PRIVATE: Only migrate pages
     * that are device-private (i.e., our VRAM pages).
     */
    args.vma = vmf->vma;
    args.src = &src_pfns;
    args.dst = &dst_pfns;
    args.start = vmf->address & PAGE_MASK;
    args.end = args.start + PAGE_SIZE;
    args.pgmap_owner = dev;
    args.flags = MIGRATE_VMA_SELECT_DEVICE_PRIVATE;
    
    /*
     * Phase 1: Setup
     *
     * Walk page tables and prepare for migration.
     * Fills src_pfns with current mapping info.
     */
    ret = migrate_vma_setup(&args);
    if (ret) {
        ps_warn(dev, "migrate_vma_setup failed: %d\n", ret);
        unlock_page(sys_page);
        put_page(sys_page);
        return VM_FAULT_SIGBUS;
    }
    
    /* Check if the page can actually be migrated */
    if (!(src_pfns & MIGRATE_PFN_MIGRATE)) {
        ps_dbg(dev, "page not migratable (src_pfns=0x%lx)\n", src_pfns);
        migrate_vma_finalize(&args);
        unlock_page(sys_page);
        put_page(sys_page);
        return 0;  /* Let kernel retry */
    }
    
    /*
     * Phase 2: Copy data from VRAM to system RAM
     *
     * This is the actual data transfer. We use our optimized
     * assembly copy routine for best performance.
     */
    vram_addr = ps_page_to_vram_addr(dev, device_page);
    sys_addr = kmap_local_page(sys_page);
    
    /* Use optimized copy from VRAM */
    ps_memcpy_from_vram(sys_addr, vram_addr, PAGE_SIZE);
    
    kunmap_local(sys_addr);
    
    /* Set up destination PFN */
    dst_pfns = migrate_pfn(page_to_pfn(sys_page));
    if (src_pfns & MIGRATE_PFN_WRITE)
        dst_pfns |= MIGRATE_PFN_WRITE;
    
    /*
     * Phase 3: Commit
     *
     * Atomically update page tables to point to system page.
     * The old VRAM page is released.
     */
    migrate_vma_pages(&args);
    
    /*
     * Phase 4: Finalize
     *
     * Clean up migration state. The VRAM page will have its
     * refcount dropped, triggering ps_page_free().
     */
    migrate_vma_finalize(&args);
    
    ps_dbg(dev, "migrate_to_ram: completed\n");
    
    return 0;
}

/*
 * Device pagemap operations
 *
 * These callbacks integrate with the kernel's device memory
 * management infrastructure.
 */
const struct dev_pagemap_ops ps_devmem_ops = {
    .page_free = ps_page_free,
    .migrate_to_ram = ps_migrate_to_ram,
};

/*
 * ps_migrate_to_device - Migrate page from system RAM to VRAM
 * @ctx: Process context
 * @start: Start address
 * @size: Size in bytes
 *
 * Explicit migration of pages TO device. This is optional -
 * pages can also migrate on first GPU access (if we had GPU
 * page fault handling).
 *
 * Returns: 0 on success, negative error code on failure
 */
int ps_migrate_to_device(struct ps_context *ctx,
                         unsigned long start,
                         unsigned long size)
{
    struct ps_device *dev = ctx->dev;
    unsigned long *src_pfns, *dst_pfns;
    struct migrate_vma args = { 0 };
    struct vm_area_struct *vma;
    unsigned long npages;
    unsigned long i;
    int ret = 0;
    
    npages = (size + PAGE_SIZE - 1) >> PAGE_SHIFT;
    if (npages > 64)
        npages = 64;  /* Process in chunks */
    
    src_pfns = kcalloc(npages, sizeof(*src_pfns), GFP_KERNEL);
    dst_pfns = kcalloc(npages, sizeof(*dst_pfns), GFP_KERNEL);
    if (!src_pfns || !dst_pfns) {
        ret = -ENOMEM;
        goto out_free;
    }

    ps_dbg(dev, "migrate_to_device: start=0x%lx size=%lu pages=%lu\n",
           start, size, npages);
    
    /* Lock the address space */
    mmap_read_lock(ctx->mm);
    
    vma = vma_lookup(ctx->mm, start);
    if (!vma) {
        ret = -EFAULT;
        goto out_unlock;
    }
    
    /* Set up migration */
    args.vma = vma;
    args.src = src_pfns;
    args.dst = dst_pfns;
    args.start = start & PAGE_MASK;
    args.end = args.start + (npages << PAGE_SHIFT);
    args.pgmap_owner = dev;
    args.flags = MIGRATE_VMA_SELECT_SYSTEM;  /* Only system pages */
    
    ret = migrate_vma_setup(&args);
    if (ret) {
        goto out_unlock;
    }
    
    /*
     * Allocate VRAM pages and copy data
     */
    for (i = 0; i < args.npages; i++) {
        struct page *src_page, *dst_page;
        void *src_addr;
        void __iomem *dst_addr;
        
        dst_pfns[i] = 0;
        
        /* Check if this page should be migrated */
        if (!(src_pfns[i] & MIGRATE_PFN_MIGRATE))
            continue;
        
        /* Allocate VRAM page */
        dst_page = ps_pool_alloc(dev->pool);
        if (!dst_page)
            continue;
        
        src_page = migrate_pfn_to_page(src_pfns[i]);
        if (!src_page) {
            ps_pool_free(dev->pool, dst_page);
            continue;
        }
        
        /* Copy data to VRAM */
        src_addr = kmap_local_page(src_page);
        dst_addr = ps_page_to_vram_addr(dev, dst_page);
        
        ps_memcpy_to_vram(dst_addr, src_addr, PAGE_SIZE);
        
        kunmap_local(src_addr);
        
        /* Lock the destination page */
        lock_page(dst_page);
        
        /* Set up destination PFN */
        dst_pfns[i] = migrate_pfn(page_to_pfn(dst_page));
        if (src_pfns[i] & MIGRATE_PFN_WRITE)
            dst_pfns[i] |= MIGRATE_PFN_WRITE;
        
        ps_stats_inc(dev, PS_STAT_MIGRATIONS_TO_DEV);
    }
    
    /* Commit migration */
    migrate_vma_pages(&args);
    migrate_vma_finalize(&args);
    
    mmap_read_unlock(ctx->mm);

    ps_dbg(dev, "migrate_to_device: completed\n");

    kfree(dst_pfns);
    kfree(src_pfns);
    return 0;

out_unlock:
    mmap_read_unlock(ctx->mm);
out_free:
    kfree(dst_pfns);
    kfree(src_pfns);
    return ret;
}
