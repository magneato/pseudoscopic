// SPDX-License-Identifier: GPL-2.0
/*
 * engine.c - GPU DMA engine interface
 *
 * Provides DMA-based page copy operations when available.
 * Falls back to programmed I/O (PIO) via our assembly
 * routines when DMA is not available or fails.
 *
 * Note: Full DMA engine support requires GPU-specific
 * initialization that varies by architecture. This is
 * a simplified implementation that focuses on correctness.
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#include <linux/dma-mapping.h>
#include <linux/highmem.h>

#include <pseudoscopic/pseudoscopic.h>
#include <pseudoscopic/asm.h>

/* DMA engine state */
struct ps_dma_engine {
    struct ps_device *dev;
    bool initialized;
    bool use_pio;  /* Fallback to programmed I/O */
};

static struct ps_dma_engine *dma_engine;

/*
 * ps_dma_init - Initialize DMA engine
 * @dev: Device to initialize
 *
 * Sets up DMA capabilities. For Tesla GPUs, the DMA engine
 * is part of the copy engine (CE) subsystem.
 *
 * Returns: 0 on success, negative error on failure
 */
int ps_dma_init(struct ps_device *dev)
{
    int ret;
    
    dma_engine = kzalloc(sizeof(*dma_engine), GFP_KERNEL);
    if (!dma_engine)
        return -ENOMEM;
    
    dma_engine->dev = dev;
    
    /*
     * Set up DMA mask.
     *
     * Tesla GPUs support 64-bit DMA addressing, but we need
     * to verify the system supports it.
     */
    ret = dma_set_mask_and_coherent(&dev->pdev->dev, DMA_BIT_MASK(64));
    if (ret) {
        ps_warn(dev, "DMA: 64-bit mask failed, trying 32-bit\n");
        ret = dma_set_mask_and_coherent(&dev->pdev->dev, DMA_BIT_MASK(32));
        if (ret) {
            ps_warn(dev, "DMA: 32-bit mask failed, using PIO\n");
            dma_engine->use_pio = true;
            dma_engine->initialized = true;
            return 0;  /* Not an error - PIO fallback */
        }
    }
    
    /*
     * Full DMA engine initialization would:
     *   1. Map GPU's copy engine registers
     *   2. Initialize command submission structures
     *   3. Set up completion interrupts
     *
     * This is complex and GPU-specific. For now, we use
     * synchronous BAR-based copy which is still efficient
     * due to write-combining and our assembly optimizations.
     */
    
    dma_engine->use_pio = true;  /* Use PIO for simplicity */
    dma_engine->initialized = true;
    
    ps_info(dev, "DMA: initialized (mode=%s)\n",
            dma_engine->use_pio ? "PIO" : "DMA");
    
    return 0;
}

/*
 * ps_dma_shutdown - Shutdown DMA engine
 * @dev: Device to shutdown
 */
void ps_dma_shutdown(struct ps_device *dev)
{
    if (!dma_engine)
        return;
    
    ps_info(dev, "DMA: shutting down\n");
    
    dma_engine->initialized = false;
    kfree(dma_engine);
    dma_engine = NULL;
}

/*
 * ps_dma_copy_to_vram - Copy page to VRAM
 * @dev: Device owning VRAM
 * @dst: Destination page (in VRAM)
 * @src: Source page (in system RAM)
 *
 * Copies a page from system RAM to VRAM. Uses DMA if
 * available, otherwise falls back to PIO.
 *
 * Returns: 0 on success, negative error on failure
 */
int ps_dma_copy_to_vram(struct ps_device *dev,
                        struct page *dst,
                        struct page *src)
{
    void __iomem *vram_addr;
    void *sys_addr;
    
    if (!dma_engine || !dma_engine->initialized) {
        ps_stats_inc(dev, PS_STAT_DMA_FAIL);
        return -ENODEV;
    }
    
    /* Get addresses */
    vram_addr = ps_page_to_vram_addr(dev, dst);
    if (!vram_addr)
        return -EINVAL;
    
    sys_addr = kmap_local_page(src);
    if (!sys_addr)
        return -ENOMEM;
    
    if (dma_engine->use_pio) {
        /*
         * PIO path: Use optimized assembly copy.
         *
         * Our ps_memcpy_to_vram() uses non-temporal stores
         * through the write-combining mapped BAR, achieving
         * near-theoretical PCIe bandwidth.
         */
        ps_memcpy_to_vram(vram_addr, sys_addr, PAGE_SIZE);
    } else {
        /*
         * DMA path: Would use GPU's copy engine.
         *
         * This would involve:
         *   1. Map source page for DMA
         *   2. Submit copy command to CE
         *   3. Wait for completion
         *   4. Unmap source page
         *
         * For now, fall back to PIO.
         */
        ps_memcpy_to_vram(vram_addr, sys_addr, PAGE_SIZE);
    }
    
    kunmap_local(sys_addr);
    
    return 0;
}

/*
 * ps_dma_copy_from_vram - Copy page from VRAM
 * @dev: Device owning VRAM
 * @dst: Destination page (in system RAM)
 * @src: Source page (in VRAM)
 *
 * Copies a page from VRAM to system RAM.
 *
 * Returns: 0 on success, negative error on failure
 */
int ps_dma_copy_from_vram(struct ps_device *dev,
                          struct page *dst,
                          struct page *src)
{
    void __iomem *vram_addr;
    void *sys_addr;
    
    if (!dma_engine || !dma_engine->initialized) {
        ps_stats_inc(dev, PS_STAT_DMA_FAIL);
        return -ENODEV;
    }
    
    /* Get addresses */
    vram_addr = ps_page_to_vram_addr(dev, src);
    if (!vram_addr)
        return -EINVAL;
    
    sys_addr = kmap_local_page(dst);
    if (!sys_addr)
        return -ENOMEM;
    
    /* Use optimized copy from VRAM */
    ps_memcpy_from_vram(sys_addr, vram_addr, PAGE_SIZE);
    
    kunmap_local(sys_addr);
    
    return 0;
}

/*
 * ps_dma_copy_sync - Synchronous bulk copy
 * @dev: Device
 * @dst: Destination address (VRAM or RAM)
 * @src: Source address (VRAM or RAM)
 * @size: Bytes to copy (must be page-aligned)
 * @to_vram: Direction flag
 *
 * Performs a synchronous bulk copy. Used for large transfers
 * where setup overhead is amortized.
 *
 * Returns: 0 on success, negative error on failure
 */
int ps_dma_copy_sync(struct ps_device *dev,
                     void *dst,
                     const void *src,
                     size_t size,
                     bool to_vram)
{
    if (!dma_engine || !dma_engine->initialized)
        return -ENODEV;
    
    /* Ensure size is cache-line aligned */
    size = ps_cache_align_size(size);
    
    if (to_vram) {
        ps_memcpy_to_vram(dst, src, size);
    } else {
        ps_memcpy_from_vram(dst, src, size);
    }
    
    return 0;
}
