// SPDX-License-Identifier: GPL-2.0
/*
 * module.c - Pseudoscopic kernel module entry point
 *
 * GPU VRAM as System RAM - Reversing depth perception
 * in the memory hierarchy.
 *
 * This module exposes NVIDIA Tesla/Datacenter GPU VRAM as
 * directly CPU-addressable memory through Linux's HMM subsystem.
 *
 * Copyright (C) 2025 Neural Splines LLC
 * Author: Robert L. Sitton, Jr.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/pci.h>
#include <linux/slab.h>
#include <linux/io.h>

#include <pseudoscopic/pseudoscopic.h>
#include <pseudoscopic/hw.h>

/* Module metadata */
MODULE_LICENSE("GPL v2");
MODULE_AUTHOR("Robert L. Sitton, Jr. <robert@neuralsplines.com>");
MODULE_DESCRIPTION("GPU VRAM as System RAM via HMM");
MODULE_VERSION(PS_VERSION_STRING);

/*
 * Module Parameters
 * -----------------
 * Configurable at load time via modprobe.
 */

static int bar_index = PS_BAR_VRAM;
module_param(bar_index, int, 0444);
MODULE_PARM_DESC(bar_index, "PCIe BAR index for VRAM (default: 1)");

static unsigned long pool_size_mb;
module_param(pool_size_mb, ulong, 0444);
MODULE_PARM_DESC(pool_size_mb, "VRAM pool size in MB (0 = entire BAR)");

static int migration_threshold = 4;
module_param(migration_threshold, int, 0644);
MODULE_PARM_DESC(migration_threshold, "Pages before async migration (default: 4)");

static bool enable_dma = true;
module_param(enable_dma, bool, 0444);
MODULE_PARM_DESC(enable_dma, "Use GPU DMA engine for transfers (default: true)");

/*
 * PCI Device Table
 * ----------------
 * Supported NVIDIA Tesla/Datacenter GPUs with Large BAR.
 */
static const struct pci_device_id ps_pci_ids[] = {
    /* Tesla P100 (Pascal GP100) - Primary development target */
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_TESLA_P100_PCIE) },
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_TESLA_P100_SXM2) },
    
    /* Tesla P40 (Pascal GP102) */
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_TESLA_P40) },
    
    /* Tesla V100 (Volta GV100) */
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_TESLA_V100_PCIE) },
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_TESLA_V100_SXM2) },
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_TESLA_V100S) },
    
    /* Quadro RTX (Turing) */
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_QUADRO_RTX_8000) },
    
    /* A100 (Ampere) */
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_A100_PCIE) },
    { PCI_DEVICE(PCI_VENDOR_ID_NVIDIA, PS_DEV_A100_SXM4) },
    
    { 0, }  /* Sentinel */
};
MODULE_DEVICE_TABLE(pci, ps_pci_ids);

/*
 * Global State
 * ------------
 * Minimal global state - one device list.
 */
static LIST_HEAD(ps_devices);
static DEFINE_MUTEX(ps_devices_lock);

/*
 * Statistics Name Mapping
 * -----------------------
 * For sysfs attribute names.
 */
static const char *ps_stat_names[PS_STAT_MAX] = {
    [PS_STAT_MIGRATIONS_TO_RAM]     = "migrations_to_ram",
    [PS_STAT_MIGRATIONS_TO_DEV]     = "migrations_to_device",
    [PS_STAT_PAGE_FAULTS]           = "page_faults",
    [PS_STAT_ALLOC_FAIL]            = "alloc_failures",
    [PS_STAT_DMA_FAIL]              = "dma_failures",
    [PS_STAT_NOTIFIER_INVALIDATE]   = "notifier_invalidations",
};

/*
 * Sysfs Attributes
 * ----------------
 * Expose statistics and configuration via sysfs.
 */

static ssize_t stat_show(struct device *dev, struct device_attribute *attr,
                         char *buf)
{
    struct ps_device *psdev = dev_get_drvdata(dev);
    int i;
    
    /* Find which stat this attribute represents */
    for (i = 0; i < PS_STAT_MAX; i++) {
        if (strcmp(attr->attr.name, ps_stat_names[i]) == 0) {
            return sysfs_emit(buf, "%llu\n", ps_stats_read(psdev, i));
        }
    }
    return -EINVAL;
}

static ssize_t vram_size_show(struct device *dev, struct device_attribute *attr,
                              char *buf)
{
    struct ps_device *psdev = dev_get_drvdata(dev);
    return sysfs_emit(buf, "%llu\n", (unsigned long long)psdev->vram_size);
}

static ssize_t pool_total_show(struct device *dev, struct device_attribute *attr,
                               char *buf)
{
    struct ps_device *psdev = dev_get_drvdata(dev);
    return sysfs_emit(buf, "%lu\n", ps_pool_total(psdev->pool));
}

static ssize_t pool_free_show(struct device *dev, struct device_attribute *attr,
                              char *buf)
{
    struct ps_device *psdev = dev_get_drvdata(dev);
    return sysfs_emit(buf, "%lu\n", ps_pool_free_count(psdev->pool));
}

static ssize_t version_show(struct device *dev, struct device_attribute *attr,
                            char *buf)
{
    return sysfs_emit(buf, "%s\n", PS_VERSION_STRING);
}

/* Declare attributes */
static DEVICE_ATTR_RO(vram_size);
static DEVICE_ATTR_RO(pool_total);
static DEVICE_ATTR_RO(pool_free);
static DEVICE_ATTR_RO(version);

/* Statistics attributes - dynamically named */
static DEVICE_ATTR(migrations_to_ram, 0444, stat_show, NULL);
static DEVICE_ATTR(migrations_to_device, 0444, stat_show, NULL);
static DEVICE_ATTR(page_faults, 0444, stat_show, NULL);
static DEVICE_ATTR(alloc_failures, 0444, stat_show, NULL);
static DEVICE_ATTR(dma_failures, 0444, stat_show, NULL);
static DEVICE_ATTR(notifier_invalidations, 0444, stat_show, NULL);

static struct attribute *ps_device_attrs[] = {
    &dev_attr_vram_size.attr,
    &dev_attr_pool_total.attr,
    &dev_attr_pool_free.attr,
    &dev_attr_version.attr,
    &dev_attr_migrations_to_ram.attr,
    &dev_attr_migrations_to_device.attr,
    &dev_attr_page_faults.attr,
    &dev_attr_alloc_failures.attr,
    &dev_attr_dma_failures.attr,
    &dev_attr_notifier_invalidations.attr,
    NULL,
};

static const struct attribute_group ps_device_attr_group = {
    .name = "pseudoscopic",
    .attrs = ps_device_attrs,
};

/*
 * PCI Probe
 * ---------
 * Called when a supported device is found.
 * Initializes hardware access and registers with HMM.
 */
static int ps_pci_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
    struct ps_device *dev;
    int ret;
    
    dev_info(&pdev->dev, "pseudoscopic: probing device %04x:%04x\n",
             pdev->vendor, pdev->device);
    
    /* Allocate device structure */
    dev = devm_kzalloc(&pdev->dev, sizeof(*dev), GFP_KERNEL);
    if (!dev)
        return -ENOMEM;
    
    dev->pdev = pdev;
    spin_lock_init(&dev->lock);
    
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
        dev_err(&pdev->dev, "pseudoscopic: failed to map BAR: %d\n", ret);
        return ret;
    }
    
    dev_info(&pdev->dev, "pseudoscopic: mapped %llu MB VRAM at BAR%d\n",
             (unsigned long long)(dev->vram_size >> 20), bar_index);
    
    /* Validate BAR configuration */
    ret = ps_bar_validate(dev);
    if (ret) {
        dev_warn(&pdev->dev, "pseudoscopic: BAR validation failed: %d\n", ret);
        /* Continue anyway - may still work */
    }
    
    /* Register device memory with HMM */
    ret = ps_hmm_register(dev);
    if (ret) {
        dev_err(&pdev->dev, "pseudoscopic: failed to register with HMM: %d\n", ret);
        goto err_bar;
    }
    dev->hmm_registered = true;
    
    /* Create page pool */
    dev->pool = ps_pool_create(dev);
    if (!dev->pool) {
        dev_err(&pdev->dev, "pseudoscopic: failed to create page pool\n");
        ret = -ENOMEM;
        goto err_hmm;
    }
    
    dev_info(&pdev->dev, "pseudoscopic: created pool with %lu pages\n",
             ps_pool_total(dev->pool));
    
    /* Initialize DMA engine if requested */
    if (enable_dma) {
        ret = ps_dma_init(dev);
        if (ret) {
            dev_warn(&pdev->dev, "pseudoscopic: DMA init failed: %d (using PIO)\n", ret);
            /* Non-fatal - fall back to programmed I/O */
        }
    }
    
    /* Create sysfs attributes */
    ret = sysfs_create_group(&pdev->dev.kobj, &ps_device_attr_group);
    if (ret) {
        dev_warn(&pdev->dev, "pseudoscopic: failed to create sysfs group: %d\n", ret);
        /* Non-fatal */
    }
    
    /* Store device reference */
    pci_set_drvdata(pdev, dev);
    dev_set_drvdata(&pdev->dev, dev);
    
    /* Add to global device list */
    mutex_lock(&ps_devices_lock);
    list_add_tail(&dev->pool->dev->pool->dev->pool->dev->pdev->dev.devres_head,
                  &ps_devices);
    mutex_unlock(&ps_devices_lock);
    
    dev->initialized = true;
    
    dev_info(&pdev->dev, "pseudoscopic: device ready - %llu MB VRAM available\n",
             (unsigned long long)(dev->vram_size >> 20));
    
    return 0;

err_hmm:
    if (dev->hmm_registered)
        ps_hmm_unregister(dev);
err_bar:
    ps_bar_unmap(dev);
    return ret;
}

/*
 * PCI Remove
 * ----------
 * Called when device is removed or driver unloaded.
 * Must cleanly tear down all state.
 */
static void ps_pci_remove(struct pci_dev *pdev)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    dev_info(&pdev->dev, "pseudoscopic: removing device\n");
    
    if (!dev)
        return;
    
    dev->initialized = false;
    
    /* Remove sysfs attributes */
    sysfs_remove_group(&pdev->dev.kobj, &ps_device_attr_group);
    
    /* Shutdown DMA engine */
    ps_dma_shutdown(dev);
    
    /* Destroy page pool (waits for all pages to be freed) */
    if (dev->pool)
        ps_pool_destroy(dev->pool);
    
    /* Unregister from HMM */
    if (dev->hmm_registered)
        ps_hmm_unregister(dev);
    
    /* Unmap BAR */
    ps_bar_unmap(dev);
    
    /* Log final statistics */
    dev_info(&pdev->dev, "pseudoscopic: final stats - "
             "to_ram=%llu to_dev=%llu faults=%llu\n",
             ps_stats_read(dev, PS_STAT_MIGRATIONS_TO_RAM),
             ps_stats_read(dev, PS_STAT_MIGRATIONS_TO_DEV),
             ps_stats_read(dev, PS_STAT_PAGE_FAULTS));
    
    dev_info(&pdev->dev, "pseudoscopic: device removed\n");
}

/*
 * PCI Error Recovery
 * ------------------
 * Handle PCIe errors gracefully.
 */
static pci_ers_result_t ps_pci_error(struct pci_dev *pdev,
                                     pci_channel_state_t state)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    dev_err(&pdev->dev, "pseudoscopic: PCIe error detected, state=%d\n", state);
    
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
    dev_info(&pdev->dev, "pseudoscopic: slot reset\n");
    
    if (pcim_enable_device(pdev)) {
        dev_err(&pdev->dev, "pseudoscopic: failed to re-enable device\n");
        return PCI_ERS_RESULT_DISCONNECT;
    }
    
    pci_set_master(pdev);
    return PCI_ERS_RESULT_RECOVERED;
}

static void ps_pci_resume(struct pci_dev *pdev)
{
    struct ps_device *dev = pci_get_drvdata(pdev);
    
    dev_info(&pdev->dev, "pseudoscopic: resuming\n");
    dev->initialized = true;
}

static const struct pci_error_handlers ps_pci_error_handlers = {
    .error_detected = ps_pci_error,
    .slot_reset = ps_pci_slot_reset,
    .resume = ps_pci_resume,
};

/*
 * PCI Driver Structure
 * --------------------
 */
static struct pci_driver ps_pci_driver = {
    .name = "pseudoscopic",
    .id_table = ps_pci_ids,
    .probe = ps_pci_probe,
    .remove = ps_pci_remove,
    .err_handler = &ps_pci_error_handlers,
};

/*
 * Module Init
 * -----------
 */
static int __init ps_init(void)
{
    int ret;
    
    pr_info("pseudoscopic: GPU VRAM as System RAM v%s\n", PS_VERSION_STRING);
    pr_info("pseudoscopic: Copyright (C) 2025 Neural Splines LLC\n");
    
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
 * Module Exit
 * -----------
 */
static void __exit ps_exit(void)
{
    pr_info("pseudoscopic: unloading driver\n");
    
    pci_unregister_driver(&ps_pci_driver);
    
    pr_info("pseudoscopic: driver unloaded\n");
}

module_init(ps_init);
module_exit(ps_exit);
