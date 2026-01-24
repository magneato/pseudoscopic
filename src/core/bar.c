// SPDX-License-Identifier: GPL-2.0
/*
 * bar.c - The Airlock
 *
 * Handles mapping GPU VRAM through the PCIe BAR aperture.
 * Tesla/Datacenter GPUs ship with Large BAR enabled, exposing
 * the full VRAM. Consumer GPUs may need resizing.
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#include <linux/pci.h>
#include <linux/io.h>

#include <pseudoscopic/pseudoscopic.h>
#include <pseudoscopic/hw.h>
#include <pseudoscopic/asm.h>

/* Exported from module.c */
extern int ps_bar_index;

/*
 * ps_bar_map - Map GPU VRAM BAR with write-combining
 * @dev: Device to map
 *
 * Maps the VRAM BAR with write-combining (WC) attributes.
 * WC is essential for bulk transfer performance - it coalesces
 * writes into full cache lines before posting to PCIe.
 *
 * Returns: 0 on success, negative error code on failure
 */
int ps_bar_map(struct ps_device *dev)
{
    struct pci_dev *pdev = dev->pdev;
    resource_size_t bar_start, bar_size;
    unsigned long bar_flags;
    
    /* Get BAR information */
    bar_start = pci_resource_start(pdev, ps_bar_index);
    bar_size = pci_resource_len(pdev, ps_bar_index);
    bar_flags = pci_resource_flags(pdev, ps_bar_index);
    
    /* Validate BAR exists and is memory-mapped */
    if (!bar_start || !bar_size) {
        ps_err(dev, "BAR%d not present or empty\n", ps_bar_index);
        return -ENODEV;
    }
    
    if (!(bar_flags & IORESOURCE_MEM)) {
        ps_err(dev, "BAR%d is not memory-mapped (flags=0x%lx)\n",
               ps_bar_index, bar_flags);
        return -EINVAL;
    }
    
    ps_dbg(dev, "BAR%d: start=0x%llx size=%llu MB flags=0x%lx\n",
           ps_bar_index, (unsigned long long)bar_start,
           (unsigned long long)(bar_size >> 20), bar_flags);
    
    /* Request the region */
    if (!devm_request_mem_region(&pdev->dev, bar_start, bar_size,
                                  "pseudoscopic")) {
        ps_err(dev, "failed to request BAR%d region\n", ps_bar_index);
        return -EBUSY;
    }
    
    /*
     * Map with write-combining (WC).
     *
     * pci_iomap_wc() sets pgprot_writecombine() which:
     *   - Allows write coalescing (multiple writes → one PCIe packet)
     *   - Avoids read-for-ownership overhead
     *   - Keeps data out of CPU cache
     *
     * This is critical for achieving near-theoretical PCIe bandwidth.
     */
    dev->vram = pci_iomap_wc(pdev, ps_bar_index, 0);  /* 0 = map entire BAR */
    if (!dev->vram) {
        ps_err(dev, "failed to iomap BAR%d with write-combining\n", ps_bar_index);
        return -ENOMEM;
    }
    
    dev->vram_base = bar_start;
    dev->vram_size = bar_size;
    
    ps_info(dev, "mapped %llu MB VRAM at %px (phys 0x%llx)\n",
            (unsigned long long)(bar_size >> 20),
            dev->vram, (unsigned long long)bar_start);
    
    return 0;
}

/*
 * ps_bar_unmap - Unmap GPU VRAM BAR
 * @dev: Device to unmap
 *
 * Releases the BAR mapping. Safe to call if not mapped.
 */
void ps_bar_unmap(struct ps_device *dev)
{
    if (dev->vram) {
        pci_iounmap(dev->pdev, dev->vram);
        dev->vram = NULL;
    }
    
    dev->vram_base = 0;
    dev->vram_size = 0;
}

/*
 * ps_bar_validate - Validate BAR configuration
 * @dev: Device to validate
 *
 * Performs sanity checks on the BAR mapping to catch
 * configuration issues early.
 *
 * Returns: 0 on success, negative error code on failure
 */
int ps_bar_validate(struct ps_device *dev)
{
    u32 test_val;
    
    /* Check size is reasonable */
    if (dev->vram_size < (256UL << 20)) {
        ps_warn(dev, "BAR size %llu MB seems small - Large BAR may be disabled\n",
                (unsigned long long)(dev->vram_size >> 20));
        /* Don't fail - might still work */
    }
    
    if (dev->vram_size > PS_MAX_VRAM_SIZE) {
        ps_err(dev, "BAR size %llu exceeds maximum supported\n",
               (unsigned long long)dev->vram_size);
        return -EINVAL;
    }
    
    /*
     * Basic access test - write a pattern and read it back.
     * This catches mapping failures that didn't error earlier.
     */
    writel(0xDEADBEEF, dev->vram);
    ps_sfence();  /* Ensure write completes */
    test_val = readl(dev->vram);
    
    if (test_val != 0xDEADBEEF) {
        ps_err(dev, "BAR access test failed: wrote 0xDEADBEEF, read 0x%08x\n",
               test_val);
        return -EIO;
    }
    
    /* Clear test pattern */
    writel(0, dev->vram);
    ps_sfence();
    
    ps_dbg(dev, "BAR validation passed\n");
    return 0;
}

/*
 * ps_bar_resize - Attempt to resize BAR (for consumer GPUs)
 * @dev: Device to resize
 * @size: Desired size in bytes
 *
 * Consumer GPUs often default to 256MB BAR aperture. This
 * function attempts to resize using PCIe resizable BAR
 * capability. Requires BIOS support.
 *
 * Returns: 0 on success, negative error code on failure
 */
int ps_bar_resize(struct ps_device *dev, resource_size_t size)
{
    struct pci_dev *pdev = dev->pdev;
    int pos;
    u32 cap, ctrl;
    unsigned int target_size_code;
    
    /* Find resizable BAR capability */
    pos = pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_REBAR);
    if (!pos) {
        ps_warn(dev, "resizable BAR capability not found\n");
        return -ENODEV;
    }
    
    /* Read capability to check supported sizes */
    pci_read_config_dword(pdev, pos + PCI_REBAR_CAP, &cap);
    
    /* Convert desired size to BAR size code */
    target_size_code = bytes_to_rebar_size(size);
    
    ps_dbg(dev, "resizable BAR: cap=0x%08x, target size code=%u\n",
           cap, target_size_code);
    
    /* Check if target size is supported */
    if (!(cap & (1 << target_size_code))) {
        ps_warn(dev, "BAR size code %u not supported (cap=0x%08x)\n",
                target_size_code, cap);
        return -EINVAL;
    }
    
    /*
     * Resize procedure:
     * 1. Disable memory decode
     * 2. Release BAR resource
     * 3. Set new size in control register
     * 4. Reassign BAR resource
     * 5. Re-enable memory decode
     */
    
    /* This is a simplified version - full implementation would
     * need to coordinate with PCI subsystem more carefully */
    
    pci_read_config_dword(pdev, pos + PCI_REBAR_CTRL, &ctrl);
    ctrl &= ~PCI_REBAR_CTRL_BAR_SIZE_MASK;
    ctrl |= (target_size_code << PCI_REBAR_CTRL_BAR_SIZE_SHIFT);
    ctrl |= (bar_index & PCI_REBAR_CTRL_BAR_IDX_MASK);
    
    ps_info(dev, "attempting BAR resize to %llu MB (code=%u)\n",
            (unsigned long long)(size >> 20), target_size_code);
    
    /* Note: Actual resize requires more coordination with PCI core.
     * For now, we just report if it would be possible. */
    ps_warn(dev, "BAR resize not fully implemented - reboot with kernel param "
            "pci=realloc or BIOS setting\n");
    
    return -ENOSYS;  /* Not implemented */
}

/*
 * ps_bar_read_mmio - Read from MMIO (BAR0)
 * @dev: Device to read from
 * @offset: Register offset
 *
 * Helper for reading GPU registers. Not commonly needed
 * for basic VRAM access.
 *
 * Returns: Register value
 */
u32 ps_bar_read_mmio(struct ps_device *dev, u32 offset)
{
    void __iomem *mmio;
    u32 val;
    
    /* Map BAR0 temporarily for MMIO access */
    mmio = pci_iomap(dev->pdev, PS_BAR_MMIO, 0);
    if (!mmio)
        return 0xFFFFFFFF;
    
    val = readl(mmio + offset);
    pci_iounmap(dev->pdev, mmio);
    
    return val;
}

/*
 * ps_bar_get_gpu_arch - Identify GPU architecture
 * @dev: Device to identify
 *
 * Reads boot register to determine GPU generation.
 * Useful for enabling architecture-specific features.
 *
 * Returns: Architecture code (NV_PMC_BOOT_0_ARCH_*)
 */
unsigned int ps_bar_get_gpu_arch(struct ps_device *dev)
{
    u32 boot0;
    
    boot0 = ps_bar_read_mmio(dev, NV_PMC_BOOT_0);
    if (boot0 == 0xFFFFFFFF)
        return 0;
    
    return nv_arch_from_boot0(boot0);
}
