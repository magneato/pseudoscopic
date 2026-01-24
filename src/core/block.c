// SPDX-License-Identifier: GPL-2.0
/*
 * block.c - The Devouring Maw
 *
 * Implements a Multi-Queue (blk-mq) block device interface for VRAM.
 * Routes IO requests directly to ps_memcpy_wc for maximum throughput.
 *
 * Exposes VRAM as:
 *   - /dev/psdiskN in ramdisk mode (general purpose)
 *   - /dev/pswapN in swap mode (optimized for page-sized I/O)
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#include <linux/blk-mq.h>
#include <linux/blkdev.h>
#include <linux/hdreg.h>
#include <linux/bio.h>
#include <linux/fs.h>
#include <linux/highmem.h>

#include <pseudoscopic/pseudoscopic.h>
#include <pseudoscopic/asm.h>

/*
 * ps_block_queue_rq - Handle an I/O request
 * @hctx: Hardware context
 * @bd:   Queue data containing the request
 *
 * The kernel calls this when it wants to read/write sectors.
 * We map the scatter-gather list and fire our ASM routines.
 *
 * Returns: BLK_STS_OK on success, error status on failure
 */
static blk_status_t ps_block_queue_rq(struct blk_mq_hw_ctx *hctx,
                                      const struct blk_mq_queue_data *bd)
{
    struct request *rq = bd->rq;
    struct ps_device *dev = rq->q->queuedata;
    struct bio_vec bvec;
    struct req_iterator iter;
    void __iomem *vram_addr;
    void *sys_addr;
    loff_t pos;
    blk_status_t status = BLK_STS_OK;

    blk_mq_start_request(rq);

    /* Sanity check */
    if (!dev || !dev->vram || dev->dying) {
        status = BLK_STS_IOERR;
        goto done;
    }

    /* Get VRAM offset from sector request (512-byte sectors) */
    pos = blk_rq_pos(rq) << 9;

    /* Verify bounds */
    if (pos + blk_rq_bytes(rq) > dev->vram_size) {
        ps_warn(dev, "block: request out of bounds (pos=%lld, size=%u)\n",
                (long long)pos, blk_rq_bytes(rq));
        status = BLK_STS_IOERR;
        goto done;
    }

    vram_addr = dev->vram + pos;

    /*
     * Iterate over memory segments in the request.
     * Use ASM-optimized copy for each segment.
     * 
     * The beauty here: blk-mq coalesces adjacent requests,
     * and our ASM routines are optimized for bulk transfer.
     */
    rq_for_each_segment(bvec, rq, iter) {
        size_t len = bvec.bv_len;
        
        sys_addr = kmap_local_page(bvec.bv_page) + bvec.bv_offset;

        if (rq_data_dir(rq) == WRITE) {
            /*
             * CPU RAM → VRAM (Write)
             * Using non-temporal stores via ps_memcpy_to_vram
             */
            ps_memcpy_to_vram(vram_addr, sys_addr, len);
            ps_stats_inc(dev, PS_STAT_BLOCK_WRITE);
        } else {
            /*
             * VRAM → CPU RAM (Read)
             * Using temporal stores via ps_memcpy_from_vram
             */
            ps_memcpy_from_vram(sys_addr, vram_addr, len);
            ps_stats_inc(dev, PS_STAT_BLOCK_READ);
        }

        kunmap_local(sys_addr);
        vram_addr += len;
    }

done:
    blk_mq_end_request(rq, status);
    return BLK_STS_OK;
}

static const struct blk_mq_ops ps_mq_ops = {
    .queue_rq = ps_block_queue_rq,
};

/*
 * ps_block_init - Birth of the Block Device
 * @dev: Pseudoscopic device to expose
 *
 * Creates the block device infrastructure and registers
 * it with the kernel.
 *
 * Returns: 0 on success, negative error on failure
 */
int ps_block_init(struct ps_device *dev)
{
    struct ps_block_dev *bdev;
    struct gendisk *disk;
    int ret;

    bdev = kzalloc(sizeof(*bdev), GFP_KERNEL);
    if (!bdev)
        return -ENOMEM;

    spin_lock_init(&bdev->lock);
    dev->bdev = bdev;

    /*
     * 1. Initialize Tag Set
     *
     * blk-mq uses "tags" to track in-flight requests.
     * We use a single hardware queue since VRAM access
     * is fundamentally serialized through the PCIe BAR.
     */
    bdev->tag_set.ops = &ps_mq_ops;
    bdev->tag_set.nr_hw_queues = 1;
    bdev->tag_set.queue_depth = 128;
    bdev->tag_set.numa_node = NUMA_NO_NODE;
    bdev->tag_set.cmd_size = 0;
    bdev->tag_set.flags = BLK_MQ_F_SHOULD_MERGE;

    ret = blk_mq_alloc_tag_set(&bdev->tag_set);
    if (ret) {
        ps_err(dev, "block: failed to allocate tag set: %d\n", ret);
        goto err_free;
    }

    /*
     * 2. Allocate Disk
     *
     * blk_mq_alloc_disk creates the gendisk and associates
     * it with our tag set.
     */
    disk = blk_mq_alloc_disk(&bdev->tag_set, NULL, dev);
    if (IS_ERR(disk)) {
        ret = PTR_ERR(disk);
        ps_err(dev, "block: failed to allocate disk: %d\n", ret);
        goto err_tags;
    }

    /*
     * 3. Configure Disk
     */
    bdev->disk = disk;
    bdev->queue = disk->queue;

    /* 
     * Naming convention:
     *   /dev/psdisk0, psdisk1, ... for ramdisk mode
     *   /dev/pswap0, pswap1, ...   for swap mode
     */
    snprintf(disk->disk_name, 32, "ps%s%d",
             (dev->mode == PS_MODE_SWAP) ? "wap" : "disk",
             dev->id);

    disk->major = 0;        /* Auto-allocate major number */
    disk->first_minor = 0;
    disk->minors = 1;
    disk->fops = NULL;      /* No special ioctls needed */
    disk->private_data = dev;

    /* Capacity in 512-byte sectors */
    set_capacity(disk, dev->vram_size >> 9);

    /*
     * 4. Queue Limits Optimization
     *
     * Align to cache lines (64 bytes) for our ASM routines.
     * Use 4KB blocks for general efficiency.
     */
    blk_queue_physical_block_size(bdev->queue, 4096);
    blk_queue_logical_block_size(bdev->queue, 4096);
    blk_queue_io_min(bdev->queue, 64);
    blk_queue_io_opt(bdev->queue, 4096 * 4);  /* 16KB optimal */

    /*
     * Mark as non-rotational (SSD-like) for better scheduling.
     * Disable read-ahead since PCIe latency dominates.
     */
    blk_queue_flag_set(QUEUE_FLAG_NONROT, bdev->queue);
    blk_queue_flag_clear(QUEUE_FLAG_ADD_RANDOM, bdev->queue);

    /* Store back-reference for queuedata */
    bdev->queue->queuedata = dev;

    /*
     * 5. Activate
     */
    ret = add_disk(disk);
    if (ret) {
        ps_err(dev, "block: failed to add disk: %d\n", ret);
        goto err_disk;
    }

    ps_info(dev, "block: /dev/%s ready (%llu MB)\n",
            disk->disk_name,
            (unsigned long long)(dev->vram_size >> 20));

    return 0;

err_disk:
    put_disk(disk);
err_tags:
    blk_mq_free_tag_set(&bdev->tag_set);
err_free:
    kfree(bdev);
    dev->bdev = NULL;
    return ret;
}

/*
 * ps_block_cleanup - Death of the Block Device
 * @dev: Device to cleanup
 *
 * Unregisters the block device and frees resources.
 */
void ps_block_cleanup(struct ps_device *dev)
{
    struct ps_block_dev *bdev = dev->bdev;

    if (!bdev)
        return;

    ps_info(dev, "block: removing /dev/%s\n", bdev->disk->disk_name);

    del_gendisk(bdev->disk);
    put_disk(bdev->disk);
    blk_mq_free_tag_set(&bdev->tag_set);

    kfree(bdev);
    dev->bdev = NULL;
}
