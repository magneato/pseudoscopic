// SPDX-License-Identifier: GPL-2.0
/*
 * pool.c - The Abyssal Reservoir
 *
 * Manages the allocation of VRAM pages. Since VRAM is treated as
 * a flat array of 4KB pages, we use a simple but efficient
 * bitmap allocator protected by a spinlock.
 *
 * Why bitmap instead of free list?
 *   - O(1) memory overhead per page (1 bit vs 8+ bytes)
 *   - Better cache locality for allocation searches
 *   - Trivial leak detection (just count set bits)
 *   - Easy bulk allocation via bitmap operations
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#include <linux/bitmap.h>
#include <linux/slab.h>
#include <linux/mm.h>
#include <linux/spinlock.h>

#include <pseudoscopic/pseudoscopic.h>

/*
 * ps_pool_init - Initialize the page pool
 * @dev: Device owning the pool
 *
 * Allocates and initializes the bitmap allocator.
 * Must be called before HMM registration to have the pool
 * ready for page allocation.
 *
 * Returns: 0 on success, negative error on failure
 */
int ps_pool_init(struct ps_device *dev)
{
    struct ps_pool *pool;
    unsigned long nr_pages;
    unsigned long bitmap_size;

    nr_pages = dev->vram_size >> PAGE_SHIFT;

    ps_dbg(dev, "pool: initializing for %lu pages (%llu MB)\n",
           nr_pages, (unsigned long long)(dev->vram_size >> 20));

    pool = kzalloc(sizeof(*pool), GFP_KERNEL);
    if (!pool)
        return -ENOMEM;

    /* 
     * Allocate bitmap - one bit per page
     * BITS_TO_LONGS rounds up to full unsigned longs
     */
    bitmap_size = BITS_TO_LONGS(nr_pages) * sizeof(unsigned long);
    pool->bitmap = kzalloc(bitmap_size, GFP_KERNEL);
    if (!pool->bitmap) {
        kfree(pool);
        return -ENOMEM;
    }

    spin_lock_init(&pool->lock);
    pool->nr_pages = nr_pages;
    pool->free_pages = nr_pages;

    dev->pool = pool;

    ps_info(dev, "pool: initialized %lu pages (%lu KB bitmap)\n",
            nr_pages, bitmap_size >> 10);

    return 0;
}

/*
 * ps_pool_destroy - Destroy the page pool
 * @dev: Device owning the pool
 *
 * Releases pool resources. Warns if pages are still allocated
 * (indicates a leak).
 */
void ps_pool_destroy(struct ps_device *dev)
{
    struct ps_pool *pool = dev->pool;
    unsigned long leaked;

    if (!pool)
        return;

    /* Check for leaks */
    leaked = pool->nr_pages - pool->free_pages;
    if (leaked > 0) {
        ps_warn(dev, "pool: destroying with %lu pages still allocated!\n",
                leaked);
    }

    ps_info(dev, "pool: destroyed (%lu/%lu pages were free)\n",
            pool->free_pages, pool->nr_pages);

    kfree(pool->bitmap);
    kfree(pool);
    dev->pool = NULL;
}

/*
 * ps_pool_alloc - Allocate a page from the pool
 * @pool: Pool to allocate from
 *
 * Finds and marks a free page in the bitmap.
 * Returns the struct page pointer for the allocated page.
 *
 * The returned page's PFN is: dev->pfn_first + bit_index
 *
 * Returns: Page pointer, or NULL if pool exhausted
 */
struct page *ps_pool_alloc(struct ps_pool *pool)
{
    unsigned long bit;
    struct page *page = NULL;
    struct ps_device *dev;
    unsigned long flags;

    if (!pool)
        return NULL;

    spin_lock_irqsave(&pool->lock, flags);

    if (pool->free_pages == 0) {
        spin_unlock_irqrestore(&pool->lock, flags);
        return NULL;
    }

    /* Find first zero bit (free page) */
    bit = find_first_zero_bit(pool->bitmap, pool->nr_pages);
    if (bit >= pool->nr_pages) {
        /* Shouldn't happen if free_pages > 0, but be defensive */
        spin_unlock_irqrestore(&pool->lock, flags);
        return NULL;
    }

    /* Mark as allocated */
    __set_bit(bit, pool->bitmap);
    pool->free_pages--;

    spin_unlock_irqrestore(&pool->lock, flags);

    /*
     * Convert bit index to struct page
     *
     * This requires HMM registration to have completed first,
     * as pfn_to_page() needs valid struct page entries.
     */
    dev = container_of(&pool, struct ps_device, pool);
    
    /* 
     * Walk back from pool to device via container_of
     * The pool pointer is stored in dev->pool, so we need
     * to find the device that owns this pool differently.
     */
    page = pfn_to_page(dev->pfn_first + bit);

    return page;
}

/*
 * ps_pool_free - Return a page to the pool
 * @pool: Pool to return to
 * @page: Page to free
 *
 * Marks the page as free in the bitmap.
 * Detects double-free errors.
 */
void ps_pool_free(struct ps_pool *pool, struct page *page)
{
    struct ps_device *dev;
    unsigned long pfn;
    unsigned long bit;
    unsigned long flags;

    if (!pool || !page)
        return;

    /* Get device from page's pagemap owner */
    dev = ps_page_to_dev(page);
    if (!dev) {
        pr_err("pseudoscopic: ps_pool_free: page has no device owner\n");
        return;
    }

    /* Calculate bit index from PFN */
    pfn = page_to_pfn(page);
    if (pfn < dev->pfn_first || pfn >= dev->pfn_last) {
        ps_err(dev, "pool: freeing page with invalid PFN %lu\n", pfn);
        return;
    }

    bit = pfn - dev->pfn_first;

    spin_lock_irqsave(&pool->lock, flags);

    /* Double-free detection */
    if (!test_bit(bit, pool->bitmap)) {
        ps_warn(dev, "pool: double free detected at bit %lu (PFN %lu)\n",
                bit, pfn);
    } else {
        __clear_bit(bit, pool->bitmap);
        pool->free_pages++;
    }

    spin_unlock_irqrestore(&pool->lock, flags);
}

/*
 * ps_pool_total - Get total pages in pool
 * @pool: Pool to query
 *
 * Returns: Total page count (capacity)
 */
unsigned long ps_pool_total(struct ps_pool *pool)
{
    return pool ? pool->nr_pages : 0;
}

/*
 * ps_pool_free_count - Get free pages in pool
 * @pool: Pool to query
 *
 * Returns: Number of unallocated pages
 */
unsigned long ps_pool_free_count(struct ps_pool *pool)
{
    return pool ? pool->free_pages : 0;
}

/*
 * ps_pool_alloc_range - Allocate contiguous pages
 * @pool: Pool to allocate from
 * @count: Number of contiguous pages needed
 * @start_bit: Output - starting bit index
 *
 * Finds a contiguous range of free pages.
 * Useful for hugepage or DMA buffer allocation.
 *
 * Returns: First page in range, or NULL if not available
 */
static __maybe_unused struct page *ps_pool_alloc_range(struct ps_pool *pool,
                                  unsigned long count,
                                  unsigned long *start_bit)
{
    unsigned long bit;
    unsigned long flags;
    struct ps_device *dev;

    if (!pool || count == 0 || count > pool->nr_pages)
        return NULL;

    spin_lock_irqsave(&pool->lock, flags);

    if (pool->free_pages < count) {
        spin_unlock_irqrestore(&pool->lock, flags);
        return NULL;
    }

    /* Find contiguous zero bits */
    bit = bitmap_find_next_zero_area(pool->bitmap, pool->nr_pages,
                                     0, count, 0);
    if (bit >= pool->nr_pages) {
        spin_unlock_irqrestore(&pool->lock, flags);
        return NULL;
    }

    /* Mark range as allocated */
    bitmap_set(pool->bitmap, bit, count);
    pool->free_pages -= count;

    spin_unlock_irqrestore(&pool->lock, flags);

    if (start_bit)
        *start_bit = bit;

    dev = container_of(&pool, struct ps_device, pool);
    return pfn_to_page(dev->pfn_first + bit);
}

/*
 * ps_pool_free_range - Free contiguous pages
 * @pool: Pool to return to
 * @page: First page in range
 * @count: Number of pages to free
 */
static __maybe_unused void ps_pool_free_range(struct ps_pool *pool, struct page *page,
                        unsigned long count)
{
    struct ps_device *dev;
    unsigned long pfn;
    unsigned long bit;
    unsigned long flags;

    if (!pool || !page || count == 0)
        return;

    dev = ps_page_to_dev(page);
    if (!dev)
        return;

    pfn = page_to_pfn(page);
    if (pfn < dev->pfn_first || pfn + count > dev->pfn_last)
        return;

    bit = pfn - dev->pfn_first;

    spin_lock_irqsave(&pool->lock, flags);
    bitmap_clear(pool->bitmap, bit, count);
    pool->free_pages += count;
    spin_unlock_irqrestore(&pool->lock, flags);
}
