// SPDX-License-Identifier: GPL-2.0
/*
 * notifier.c - MMU notifier integration
 *
 * Provides callbacks when CPU page tables change, allowing
 * the driver to maintain coherency between CPU and device
 * state. Essential for correct operation when pages are
 * unmapped, migrated, or permissions change.
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#include <linux/mmu_notifier.h>
#include <linux/sched/mm.h>

#include <pseudoscopic/pseudoscopic.h>

/*
 * ps_notifier_invalidate - MMU invalidation callback
 * @mni: Interval notifier that triggered
 * @range: Address range being invalidated
 * @cur_seq: Current sequence number
 *
 * Called by the kernel BEFORE page table entries are modified.
 * We must ensure any device state referencing these pages is
 * consistent before returning.
 *
 * Returns: true to allow invalidation, false to block (rare)
 */
static bool ps_notifier_invalidate(struct mmu_interval_notifier *mni,
                                   const struct mmu_notifier_range *range,
                                   unsigned long cur_seq)
{
    struct ps_context *ctx = container_of(mni, struct ps_context, notifier);
    struct ps_device *dev = ctx->dev;
    
    ps_stats_inc(dev, PS_STAT_NOTIFIER_INVALIDATE);
    
    ps_dbg(dev, "notifier: invalidate 0x%lx - 0x%lx event=%d\n",
           range->start, range->end, range->event);
    
    /*
     * Skip invalidation for our own migrations.
     *
     * When we migrate pages via migrate_vma_*(), the kernel
     * sends us an invalidation notification. We don't need
     * to do anything for our own operations.
     */
    if (range->event == MMU_NOTIFY_MIGRATE &&
        range->owner == dev) {
        ps_dbg(dev, "notifier: skipping self-migration\n");
        return true;
    }
    
    /*
     * Update sequence number.
     *
     * This is used by migration code to detect concurrent
     * page table changes. If the sequence number changes
     * during migration, we retry.
     */
    mmu_interval_set_seq(mni, cur_seq);
    
    /*
     * For real GPU drivers, we would:
     *   1. Invalidate any device page table entries
     *   2. Flush device TLB
     *   3. Wait for in-flight device operations
     *
     * Since we don't manage GPU page tables (we use BAR-based
     * access), we just ensure memory operations complete.
     */
    ps_mfence();  /* Full memory barrier */
    
    return true;  /* Allow invalidation to proceed */
}

/*
 * MMU interval notifier operations
 */
const struct mmu_interval_notifier_ops ps_notifier_ops = {
    .invalidate = ps_notifier_invalidate,
};

/*
 * ps_notifier_register - Register notifier for address range
 * @ctx: Process context
 * @start: Start address
 * @size: Size of range
 *
 * Registers for notifications when page tables change in
 * the specified range. Must be called before migrating
 * pages in that range.
 *
 * Returns: 0 on success, negative error code on failure
 */
int ps_notifier_register(struct ps_context *ctx,
                         unsigned long start,
                         unsigned long size)
{
    struct ps_device *dev = ctx->dev;
    int ret;
    
    ps_dbg(dev, "notifier: registering 0x%lx - 0x%lx\n",
           start, start + size);
    
    ret = mmu_interval_notifier_insert(&ctx->notifier,
                                       ctx->mm,
                                       start,
                                       size,
                                       &ps_notifier_ops);
    if (ret) {
        ps_warn(dev, "notifier: failed to register: %d\n", ret);
        return ret;
    }
    
    return 0;
}

/*
 * ps_notifier_unregister - Unregister notifier
 * @ctx: Process context
 *
 * Removes the notifier registration. Must be called before
 * the process context is freed.
 */
void ps_notifier_unregister(struct ps_context *ctx)
{
    ps_dbg(ctx->dev, "notifier: unregistering\n");
    
    mmu_interval_notifier_remove(&ctx->notifier);
}

/*
 * ps_context_create - Create per-process context
 * @dev: Device
 * @mm: Process memory map
 *
 * Creates and initializes a context for tracking per-process
 * state. Each process using VRAM gets its own context.
 *
 * Returns: Context pointer, or ERR_PTR on failure
 */
struct ps_context *ps_context_create(struct ps_device *dev,
                                     struct mm_struct *mm)
{
    struct ps_context *ctx;
    
    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx)
        return ERR_PTR(-ENOMEM);
    
    ctx->dev = dev;
    ctx->mm = mm;
    mmgrab(mm);  /* Take reference on mm */
    INIT_LIST_HEAD(&ctx->node);
    
    ps_dbg(dev, "context: created for mm=%px\n", mm);
    
    return ctx;
}

/*
 * ps_context_destroy - Destroy per-process context
 * @ctx: Context to destroy
 *
 * Releases context resources. Must ensure all notifiers
 * are unregistered first.
 */
void ps_context_destroy(struct ps_context *ctx)
{
    if (!ctx)
        return;
    
    ps_dbg(ctx->dev, "context: destroying for mm=%px\n", ctx->mm);
    
    /* Release mm reference */
    if (ctx->mm)
        mmdrop(ctx->mm);
    
    kfree(ctx);
}
