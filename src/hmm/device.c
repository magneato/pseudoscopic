// SPDX-License-Identifier: GPL-2.0
/*
 * device.c - HMM device memory registration
 *
 * Registers GPU VRAM with the kernel's memory management
 * system via the Heterogeneous Memory Management (HMM)
 * subsystem. This creates struct page entries for VRAM
 * and integrates with the kernel's page allocator.
 *
 * Key concepts:
 *   - ZONE_DEVICE: A memory zone for non-standard memory
 *   - MEMORY_DEVICE_PRIVATE: Pages not directly CPU-mappable
 *   - dev_pagemap: Describes device memory to the kernel
 *   - migrate_to_ram: Callback for CPU access to device pages
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#include <linux/memremap.h>
#include <linux/mm.h>
#include <linux/pfn_t.h>
#include <linux/migrate.h>

#include <pseudoscopic/pseudoscopic.h>

/*
 * Device pagemap operations - defined in migrate.c
 */
extern const struct dev_pagemap_ops ps_devmem_ops;

/*
 * ps_hmm_register - Register VRAM as device memory
 * @dev: Device to register
 *
 * Creates struct page entries for VRAM pages and registers
 * them with the kernel's memory management. After this,
 * VRAM pages can participate in page migration via HMM.
 *
 * Memory type is MEMORY_DEVICE_PRIVATE, meaning:
 *   - Pages are NOT directly CPU-accessible
 *   - CPU access triggers migrate_to_ram callback
 *   - Enables transparent demand paging
 *
 * Returns: 0 on success, negative error code on failure
 */
int ps_hmm_register(struct ps_device *dev)
{
    void *addr;
    int ret;
    
    ps_dbg(dev, "HMM: registering %llu bytes of VRAM\n",
           (unsigned long long)dev->vram_size);
    
    /*
     * Request a free memory region for our device pages.
     *
     * devm_request_free_mem_region() finds a hole in the
     * physical address space and reserves it. This doesn't
     * correspond to actual RAM - it's a pseudo-physical
     * address range for our device memory.
     */
    dev->vram_res = devm_request_free_mem_region(&dev->pdev->dev,
                                                  &iomem_resource,
                                                  dev->vram_size);
    if (IS_ERR(dev->vram_res)) {
        ret = PTR_ERR(dev->vram_res);
        ps_err(dev, "HMM: failed to request free mem region: %d\n", ret);
        dev->vram_res = NULL;
        return ret;
    }
    
    ps_dbg(dev, "HMM: allocated pseudo-physical range 0x%llx - 0x%llx\n",
           (unsigned long long)dev->vram_res->start,
           (unsigned long long)dev->vram_res->end);
    
    /*
     * Set up the device pagemap structure.
     *
     * MEMORY_DEVICE_PRIVATE:
     *   - Pages cannot be directly mapped by CPU
     *   - Any CPU access triggers migrate_to_ram callback
     *   - This is the key to transparent demand paging
     *
     * We could use MEMORY_DEVICE_GENERIC for direct mapping,
     * but that loses the migration machinery and exposes
     * raw PCIe latency on every access.
     */
    dev->pagemap.type = MEMORY_DEVICE_PRIVATE;
    dev->pagemap.range.start = dev->vram_res->start;
    dev->pagemap.range.end = dev->vram_res->end;
    dev->pagemap.nr_range = 1;
    dev->pagemap.ops = &ps_devmem_ops;
    dev->pagemap.owner = dev;
    
    /*
     * Register with the kernel memory management.
     *
     * devm_memremap_pages() does several critical things:
     *   1. Allocates struct page for every page in range
     *   2. Marks them as ZONE_DEVICE
     *   3. Sets up PFN ↔ page mapping
     *   4. Registers our callbacks for page lifecycle
     *
     * After this, pfn_to_page() works for our VRAM PFNs.
     */
    addr = devm_memremap_pages(&dev->pdev->dev, &dev->pagemap);
    if (IS_ERR(addr)) {
        ret = PTR_ERR(addr);
        ps_err(dev, "HMM: failed to memremap pages: %d\n", ret);
        devm_release_mem_region(&dev->pdev->dev,
                                dev->vram_res->start,
                                resource_size(dev->vram_res));
        dev->vram_res = NULL;
        return ret;
    }
    
    /* Calculate PFN range for our pages */
    dev->pfn_first = dev->vram_res->start >> PAGE_SHIFT;
    dev->pfn_last = (dev->vram_res->end + 1) >> PAGE_SHIFT;
    
    ps_info(dev, "HMM: registered %lu pages (PFN 0x%lx - 0x%lx)\n",
            dev->pfn_last - dev->pfn_first,
            dev->pfn_first, dev->pfn_last);
    
    return 0;
}

/*
 * ps_hmm_unregister - Unregister VRAM from HMM
 * @dev: Device to unregister
 *
 * Releases HMM registration. Must wait for all outstanding
 * migrations to complete first.
 */
void ps_hmm_unregister(struct ps_device *dev)
{
    if (!dev->vram_res)
        return;
    
    ps_info(dev, "HMM: unregistering device memory\n");
    
    /*
     * The actual cleanup is handled by devm (device-managed)
     * resources. devm_memremap_pages() registered cleanup
     * handlers that will:
     *   1. Wait for all page references to drop
     *   2. Free struct page entries
     *   3. Release the pseudo-physical region
     *
     * We just need to trigger it by releasing our reference.
     */
    
    /* Release the memory region */
    devm_release_mem_region(&dev->pdev->dev,
                            dev->vram_res->start,
                            resource_size(dev->vram_res));
    
    dev->vram_res = NULL;
    dev->pfn_first = 0;
    dev->pfn_last = 0;
    
    ps_info(dev, "HMM: device memory unregistered\n");
}

/*
 * ps_hmm_pfn_valid - Check if PFN belongs to our VRAM
 * @dev: Device to check
 * @pfn: Page frame number
 *
 * Returns: true if PFN is within our registered range
 */
bool ps_hmm_pfn_valid(struct ps_device *dev, unsigned long pfn)
{
    return pfn >= dev->pfn_first && pfn < dev->pfn_last;
}

/*
 * ps_hmm_page_to_offset - Get VRAM offset for a page
 * @dev: Device owning the page
 * @page: Page to query
 *
 * Returns: Byte offset within VRAM
 */
unsigned long ps_hmm_page_to_offset(struct ps_device *dev, struct page *page)
{
    unsigned long pfn = page_to_pfn(page);
    
    if (!ps_hmm_pfn_valid(dev, pfn))
        return 0;
    
    return (pfn - dev->pfn_first) << PAGE_SHIFT;
}
