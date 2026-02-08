// SPDX-License-Identifier: GPL-2.0
/*
 * module.c - The Bathysphere Entry Hatch
 *
 * GPU VRAM as System RAM - Reversing depth perception
 * in the memory hierarchy.
 *
 * Handles parameter parsing, device selection, and mode switching.
 * Supports identifying as a Memory Controller or a Block Device.
 *
 * Copyright (C) 2026 Neural Splines LLC
 * Author: Robert L. Sitton, Jr.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/io.h>
#include <linux/vgaarb.h>

#include <pseudoscopic/pseudoscopic.h>
#include <pseudoscopic/hw.h>

#define DRIVER_NAME "pseudoscopic"
#define DRIVER_DESC "GPU VRAM as System RAM"

/* Module metadata */
MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Robert L. Sitton, Jr. <robert@neuralsplines.com>");
MODULE_DESCRIPTION(DRIVER_DESC);
MODULE_VERSION(PS_VERSION_STRING);

/*
 * Module Parameters - The Dive Controls
 * -------------------------------------
 * Configurable at load time via modprobe.
 */

static int bar_index = PS_BAR_VRAM;
module_param(bar_index, int, 0444);
MODULE_PARM_DESC(bar_index, "PCIe BAR index for VRAM (default: 1)");

static int device_idx = -1;
module_param(device_idx, int, 0444);
MODULE_PARM_DESC(device_idx, "Specific device index to bind (0=first). -1=Auto (skip primary).");

static char *mode_str = "ram";
module_param_named(mode, mode_str, charp, 0444);
MODULE_PARM_DESC(mode, "Operation mode: 'ram' (default), 'ramdisk', 'swap'");

static int migration_threshold = 4;
module_param(migration_threshold, int, 0644);
MODULE_PARM_DESC(migration_threshold, "Pages before async migration (default: 4)");

static bool enable_dma = true;
module_param(enable_dma, bool, 0444);
MODULE_PARM_DESC(enable_dma, "Use GPU DMA engine for transfers (default: true)");

/* Exported for other modules */
int ps_bar_index = 1;
EXPORT_SYMBOL_GPL(ps_bar_index);

/* Global device counter */
static atomic_t device_count = ATOMIC_INIT(-1);

/*
 * PCI Device Table - The Sonar Targets
 * ------------------------------------
 * We bind to NVIDIA display controllers.
 * Using class-based matching for broader compatibility.
 */
static const struct pci_device_id ps_pci_tbl[] = {
    /* NVIDIA VGA compatible controllers */
    { PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID, PCI_ANY_ID, PCI_ANY_ID,
      PCI_CLASS_DISPLAY_VGA << 8, 0xFFFF00, 0 },
    /* NVIDIA 3D controllers (Tesla/compute cards) */
    { PCI_VENDOR_ID_NVIDIA, PCI_ANY_ID, PCI_ANY_ID, PCI_ANY_ID,
      PCI_CLASS_DISPLAY_3D << 8, 0xFFFF00, 0 },
    { 0, }  /* Sentinel */
};
MODULE_DEVICE_TABLE(pci, ps_pci_tbl);

/*
 * Statistics Name Mapping - The Telemetry Labels
 * ----------------------------------------------
 * For sysfs attribute names.
 */
static const char *ps_stat_names[PS_STAT_MAX] = {
    [PS_STAT_PAGE_FAULTS]           = "page_faults",
    [PS_STAT_MIGRATIONS_TO_RAM]     = "migrations_to_ram",
    [PS_STAT_MIGRATIONS_TO_DEV]     = "migrations_to_device",
    [PS_STAT_ALLOC_FAIL]            = "alloc_failures",
    [PS_STAT_DMA_FAIL]              = "dma_failures",
    [PS_STAT_NOTIFIER_INVALIDATE]   = "notifier_invalidations",
    [PS_STAT_BLOCK_READ]            = "block_reads",
    [PS_STAT_BLOCK_WRITE]           = "block_writes",
};

/*
 * ps_get_mode - Parse mode string to enum
 */
static enum ps_op_mode ps_get_mode(void)
{
    if (!strcmp(mode_str, "ramdisk"))
        return PS_MODE_RAMDISK;
    if (!strcmp(mode_str, "swap"))
        return PS_MODE_SWAP;
    return PS_MODE_RAM;
}

/*
 * ps_mode_name - Get human-readable mode name
 */
static const char *ps_mode_name(enum ps_op_mode mode)
{
    switch (mode) {
    case PS_MODE_RAM:       return "RAM";
    case PS_MODE_RAMDISK:   return "Ramdisk";
    case PS_MODE_SWAP:      return "Swap";
    default:                return "Unknown";
    }
}

/*
 * Sysfs Attributes - The Status Lights
 * ------------------------------------
 * Expose statistics and configuration.
 */

static ssize_t stat_show(struct device *device, struct device_attribute *attr,
                         char *buf)
{
    struct ps_device *dev = dev_get_drvdata(device);
    int i;
    
    for (i = 0; i < PS_STAT_MAX; i++) {
        if (strcmp(attr->attr.name, ps_stat_names[i]) == 0)
            return sysfs_emit(buf, "%llu\n", ps_stats_read(dev, i));
    }
    return -EINVAL;
}

static ssize_t vram_size_show(struct device *device, struct device_attribute *attr,
                              char *buf)
{
    struct ps_device *dev = dev_get_drvdata(device);
    return sysfs_emit(buf, "%llu\n", (unsigned long long)dev->vram_size);
}

static ssize_t pool_free_show(struct device *device, struct device_attribute *attr,
                              char *buf)
{
    struct ps_device *dev = dev_get_drvdata(device);
    return sysfs_emit(buf, "%lu\n", ps_pool_free_count(dev->pool));
}

static ssize_t mode_show(struct device *device, struct device_attribute *attr,
                         char *buf)
{
    struct ps_device *dev = dev_get_drvdata(device);
    return sysfs_emit(buf, "%s\n", ps_mode_name(dev->mode));
}

static ssize_t version_show(struct device *device, struct device_attribute *attr,
                            char *buf)
{
    return sysfs_emit(buf, "%s\n", PS_VERSION_STRING);
}

/* Declare attributes */
static DEVICE_ATTR_RO(vram_size);
static DEVICE_ATTR_RO(pool_free);
static DEVICE_ATTR_RO(mode);
static DEVICE_ATTR_RO(version);

/* Statistics attributes */
static DEVICE_ATTR(page_faults, 0444, stat_show, NULL);
static DEVICE_ATTR(migrations_to_ram, 0444, stat_show, NULL);
static DEVICE_ATTR(migrations_to_device, 0444, stat_show, NULL);
static DEVICE_ATTR(alloc_failures, 0444, stat_show, NULL);
static DEVICE_ATTR(dma_failures, 0444, stat_show, NULL);
static DEVICE_ATTR(notifier_invalidations, 0444, stat_show, NULL);
static DEVICE_ATTR(block_reads, 0444, stat_show, NULL);
static DEVICE_ATTR(block_writes, 0444, stat_show, NULL);

static struct attribute *ps_device_attrs[] = {
    &dev_attr_vram_size.attr,
    &dev_attr_pool_free.attr,
    &dev_attr_mode.attr,
    &dev_attr_version.attr,
    &dev_attr_page_faults.attr,
    &dev_attr_migrations_to_ram.attr,
    &dev_attr_migrations_to_device.attr,
    &dev_attr_alloc_failures.attr,
    &dev_attr_dma_failures.attr,
    &dev_attr_notifier_invalidations.attr,
    &dev_attr_block_reads.attr,
    &dev_attr_block_writes.attr,
    NULL,
};

static const struct attribute_group ps_device_attr_group = {
    .name = "pseudoscopic",
    .attrs = ps_device_attrs,
};

/*
 * ps_probe - The Docking Procedure
 * --------------------------------
 * Called when a supported device is found.
 * Filters by device index and primary display protection.
 */
static int ps_probe(struct pci_dev *pdev, const struct pci_device_id *ent)
{
    struct ps_device *dev;
    int ret;
    int my_idx;
    bool is_primary;
    
    /* Increment device counter to establish ID */
    my_idx = atomic_inc_return(&device_count);
    
    dev_info(&pdev->dev, "pseudoscopic: probing device %04x:%04x (index %d)\n",
             pdev->vendor, pdev->device, my_idx);
    
    /*
     * Filter: Device Selection
     * If user asked for a specific device index, ignore others.
     */
    if (device_idx >= 0 && my_idx != device_idx) {
        ps_dbg_raw("pseudoscopic: skipping device %d (requested %d)\n",
                   my_idx, device_idx);
        return -ENODEV;
    }
    
    /*
     * Filter: Primary Display Protection
     * If using auto-selection (device_idx == -1), skip the boot VGA
     * to prevent killing the user's desktop session.
     */
    is_primary = (vga_default_device() == pdev);
    if (device_idx == -1 && is_primary) {
        dev_info(&pdev->dev, "pseudoscopic: skipping primary display "
                 "(use device_idx=N to force)\n");
        return -ENODEV;
    }
    
    /* Allocate device structure */
    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;
    
    dev->pdev = pdev;
    dev->id = my_idx;
    dev->mode = ps_get_mode();
    mutex_init(&dev->lock);
    
    /* Export bar_index for other modules */
    ps_bar_index = bar_index;
    
    /* Initialize statistics to zero */
    for (int i = 0; i < PS_STAT_MAX; i++)
        atomic64_set(&dev->stats[i], 0);
    
    /* Enable PCI device */
    ret = pcim_enable_device(pdev);
    if (ret) {
        dev_err(&pdev->dev, "pseudoscopic: failed to enable PCI device: %d\n", ret);
        return ret;
    }
    
    /* Enable bus mastering for DMA */
    pci_set_master(pdev);
    
    /* Map VRAM BAR */
    ret = ps_bar_map(dev);
    if (ret) {
        ps_err(dev, "failed to map BAR: %d\n", ret);
        return ret;
    }
    
    ps_info(dev, "mapped %llu MB VRAM at BAR%d\n",
            (unsigned long long)(dev->vram_size >> 20), bar_index);
    
    /* Validate BAR configuration */
    ret = ps_bar_validate(dev);
    if (ret) {
        ps_warn(dev, "BAR validation failed: %d (continuing anyway)\n", ret);
        /* Continue - may still work */
    }
    
    /* Initialize page pool (common to all modes) */
    ret = ps_pool_init(dev);
    if (ret) {
        ps_err(dev, "failed to initialize page pool: %d\n", ret);
        goto err_bar;
    }
    
    /* Initialize DMA engine if requested */
    if (enable_dma) {
        ret = ps_dma_init(dev);
        if (ret) {
            ps_warn(dev, "DMA init failed: %d (using PIO)\n", ret);
            /* Non-fatal - fall back to programmed I/O */
        }
    }
    
    /*
     * Mode Divergence - The Metamorphosis
     * -----------------------------------
     * RAM mode: Extend system memory via HMM
     * Block modes: Expose as disk device
     */
    if (dev->mode == PS_MODE_RAM) {
        /* RAM Mode: Integrate with kernel memory management */
        ret = ps_hmm_register(dev);
        if (ret) {
            ps_err(dev, "failed to register with HMM: %d\n", ret);
            goto err_dma;
        }
        
        ps_info(dev, "online (RAM mode) - %llu MB added to system\n",
                (unsigned long long)(dev->vram_size >> 20));
    } else {
        /* Block/Swap Mode: Expose as disk */
        ret = ps_block_init(dev);
        if (ret) {
            ps_err(dev, "failed to create block device: %d\n", ret);
            goto err_dma;
        }
        
        ps_info(dev, "online (%s mode) - %llu MB block device ready\n",
                ps_mode_name(dev->mode),
                (unsigned long long)(dev->vram_size >> 20));
    }
    
    /* Create sysfs attributes */
    ret = sysfs_create_group(&pdev->dev.kobj, &ps_device_attr_group);
    if (ret)
        ps_warn(dev, "failed to create sysfs group: %d\n", ret);
    
    /* Store device reference */
    pci_set_drvdata(pdev, dev);
    dev_set_drvdata(&pdev->dev, dev);
    
    dev->initialized = true;
    
    return 0;

err_dma:
    ps_dma_shutdown(dev);
    ps_pool_destroy(dev);
err_bar:
    ps_bar_unmap(dev);
    return ret;
}

/*
 * ps_remove - The Undocking Procedure
 * -----------------------------------
 * Called when device is removed or driver unloaded.
 * Must cleanly tear down all state in reverse order.
 */
static void ps_remove(struct pci_dev *pdev)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    if (!dev)
        return;
    
    ps_info(dev, "removing device\n");
    
    dev->dying = true;
    dev->initialized = false;
    
    /* Remove sysfs attributes */
    sysfs_remove_group(&pdev->dev.kobj, &ps_device_attr_group);
    
    /* Reverse order of initialization */
    if (dev->mode == PS_MODE_RAM) {
        ps_hmm_unregister(dev);
    } else {
        ps_block_cleanup(dev);
    }
    
    /* Shutdown DMA engine */
    ps_dma_shutdown(dev);
    
    /* Destroy page pool */
    ps_pool_destroy(dev);
    
    /* Unmap BAR */
    ps_bar_unmap(dev);
    
    /* Log final statistics */
    ps_info(dev, "final stats - faults=%llu to_ram=%llu to_dev=%llu\n",
            ps_stats_read(dev, PS_STAT_PAGE_FAULTS),
            ps_stats_read(dev, PS_STAT_MIGRATIONS_TO_RAM),
            ps_stats_read(dev, PS_STAT_MIGRATIONS_TO_DEV));
    
    ps_info(dev, "device removed\n");
}

/*
 * PCI Error Recovery - The Emergency Protocols
 * --------------------------------------------
 */
static pci_ers_result_t ps_pci_error(struct pci_dev *pdev,
                                     pci_channel_state_t state)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    ps_err(dev, "PCIe error detected, state=%d\n", state);
    
    switch (state) {
    case pci_channel_io_normal:
        return PCI_ERS_RESULT_CAN_RECOVER;
    case pci_channel_io_frozen:
        dev->initialized = false;
        return PCI_ERS_RESULT_NEED_RESET;
    case pci_channel_io_perm_failure:
        return PCI_ERS_RESULT_DISCONNECT;
    default:
        return PCI_ERS_RESULT_NONE;
    }
}

static pci_ers_result_t ps_pci_slot_reset(struct pci_dev *pdev)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    ps_info(dev, "slot reset\n");
    
    if (pcim_enable_device(pdev)) {
        ps_err(dev, "failed to re-enable device\n");
        return PCI_ERS_RESULT_DISCONNECT;
    }
    
    pci_set_master(pdev);
    return PCI_ERS_RESULT_RECOVERED;
}

static void ps_pci_resume(struct pci_dev *pdev)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    ps_info(dev, "resuming\n");
    dev->initialized = true;
}

static const struct pci_error_handlers ps_pci_error_handlers = {
    .error_detected = ps_pci_error,
    .slot_reset = ps_pci_slot_reset,
    .resume = ps_pci_resume,
};

/*
 * PCI Driver Structure
 */
static struct pci_driver ps_pci_driver = {
    .name = DRIVER_NAME,
    .id_table = ps_pci_tbl,
    .probe = ps_probe,
    .remove = ps_remove,
    .err_handler = &ps_pci_error_handlers,
};

/*
 * Module Init - The Descent Begins
 */
static int __init ps_init(void)
{
    int ret;
    
    pr_info("pseudoscopic: " DRIVER_DESC " v%s\n", PS_VERSION_STRING);
    pr_info("pseudoscopic: Copyright (C) 2026 Neural Splines LLC\n");
    pr_info("pseudoscopic: mode=%s device_idx=%d\n", mode_str, device_idx);
    
    /* Validate module parameters */
    if (bar_index < 0 || bar_index > 5) {
        pr_err("pseudoscopic: invalid bar_index %d (must be 0-5)\n", bar_index);
        return -EINVAL;
    }
    
    /* Register PCI driver */
    ret = pci_register_driver(&ps_pci_driver);
    if (ret) {
        pr_err("pseudoscopic: failed to register PCI driver: %d\n", ret);
        return ret;
    }
    
    pr_info("pseudoscopic: driver loaded successfully\n");
    return 0;
}

/*
 * Module Exit - Returning to the Surface
 */
static void __exit ps_exit(void)
{
    pr_info("pseudoscopic: unloading driver\n");
    
    pci_unregister_driver(&ps_pci_driver);
    
    pr_info("pseudoscopic: driver unloaded\n");
}

module_init(ps_init);
module_exit(ps_exit);
