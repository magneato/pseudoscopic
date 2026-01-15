// SPDX-License-Identifier: GPL-2.0
/*
 * pool.c - VRAM page pool management
 *
 * Manages the free list of VRAM pages. Simple and fast -
 * allocation is O(1) lock-protected list operation.
 *
 * Pages are pre-initialized during driver load, with struct
 * page entries created by devm_memremap_pages() as part of
 * HMM registration.
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#include <linux/slab.h>
#include <linux/mm.h>

#include <pseudoscopic/pseudoscopic.h>

/* Module parameter - declared in module.c */
extern unsigned long pool_size_mb;

/*
 * ps_pool_create - Create VRAM page pool
 * @dev: Device owning the pool
 *
 * Initializes the free list from all available VRAM pages.
 * Must be called after HMM registration (which creates the
 * struct page entries).
 *
 * Returns: Pool pointer on success, NULL on failure
 */
struct ps_pool *ps_pool_create(struct ps_device *dev)
{
    struct ps_pool *pool;
    unsigned long pfn;
    unsigned long count = 0;
    
    pool = kzalloc(sizeof(*pool), GFP_KERNEL);
    if (!pool)
        return NULL;
    
    pool->dev = dev;
    spin_lock_init(&pool->lock);
    pool->free_list = NULL;
    
    /*
     * Initialize free list from HMM-registered pages.
     *
     * The pages were created by devm_memremap_pages() during
     * HMM registration. We link them into our free list and
     * set zone_device_data to point back to our device.
     */
    for (pfn = dev->pfn_first; pfn < dev->pfn_last; pfn++) {
        struct page *page = pfn_to_page(pfn);
        
        /* Link into free list */
        page->zone_device_data = pool->free_list;
        pool->free_list = page;
        count++;
    }
    
    pool->total = count;
    atomic_long_set(&pool->free, count);
    
    ps_info(dev, "pool: initialized %lu pages (%llu MB)\n",
            count, (unsigned long long)(count * PAGE_SIZE) >> 20);
    
    return pool;
}

/*
 * ps_pool_destroy - Destroy VRAM page pool
 * @pool: Pool to destroy
 *
 * Waits for all pages to be returned, then frees the pool.
 * This ensures clean shutdown without leaked pages.
 */
void ps_pool_destroy(struct ps_pool *pool)
{
    unsigned long free_count;
    int wait_count = 0;
    const int max_wait = 100;  /* 10 seconds max */
    
    if (!pool)
        return;
    
    /* Wait for all pages to be returned to pool */
    while ((free_count = atomic_long_read(&pool->free)) < pool->total) {
        if (wait_count++ > max_wait) {
            ps_warn(pool->dev, "pool: timeout waiting for pages "
                    "(have %lu, need %lu)\n", free_count, pool->total);
            break;
        }
        ps_dbg(pool->dev, "pool: waiting for pages (%lu/%lu)\n",
               free_count, pool->total);
        msleep(100);
    }
    
    ps_info(pool->dev, "pool: destroyed (%lu/%lu pages returned)\n",
            atomic_long_read(&pool->free), pool->total);
    
    kfree(pool);
}

/*
 * ps_pool_alloc - Allocate a page from VRAM pool
 * @pool: Pool to allocate from
 *
 * Fast O(1) allocation from free list head.
 * Caller must check return value - may return NULL if exhausted.
 *
 * Returns: Page pointer, or NULL if pool exhausted
 */
struct page *ps_pool_alloc(struct ps_pool *pool)
{
    struct page *page;
    unsigned long flags;
    
    spin_lock_irqsave(&pool->lock, flags);
    
    page = pool->free_list;
    if (page) {
        pool->free_list = page->zone_device_data;
        page->zone_device_data = pool->dev;  /* Back-pointer for ps_page_to_dev */
        atomic_long_dec(&pool->free);
    }
    
    spin_unlock_irqrestore(&pool->lock, flags);
    
    if (!page)
        ps_stats_inc(pool->dev, PS_STAT_ALLOC_FAIL);
    
    return page;
}

/*
 * ps_pool_free - Return a page to VRAM pool
 * @pool: Pool to return to
 * @page: Page to return
 *
 * Fast O(1) free to list head.
 */
void ps_pool_free(struct ps_pool *pool, struct page *page)
{
    unsigned long flags;
    
    if (!pool || !page)
        return;
    
    spin_lock_irqsave(&pool->lock, flags);
    
    page->zone_device_data = pool->free_list;
    pool->free_list = page;
    atomic_long_inc(&pool->free);
    
    spin_unlock_irqrestore(&pool->lock, flags);
}

/*
 * ps_pool_total - Get total pages in pool
 * @pool: Pool to query
 *
 * Returns: Total page count
 */
unsigned long ps_pool_total(struct ps_pool *pool)
{
    return pool ? pool->total : 0;
}

/*
 * ps_pool_free_count - Get free pages in pool
 * @pool: Pool to query
 *
 * Returns: Free page count
 */
unsigned long ps_pool_free_count(struct ps_pool *pool)
{
    return pool ? atomic_long_read(&pool->free) : 0;
}

/*
 * ps_pool_alloc_bulk - Allocate multiple pages
 * @pool: Pool to allocate from
 * @pages: Array to fill with page pointers
 * @count: Number of pages to allocate
 *
 * Allocates multiple pages atomically. Either all succeed
 * or none are allocated (transactional semantics).
 *
 * Returns: Number of pages allocated (count on success, 0 on failure)
 */
unsigned long ps_pool_alloc_bulk(struct ps_pool *pool, 
                                  struct page **pages,
                                  unsigned long count)
{
    unsigned long flags;
    unsigned long i;
    
    if (!pool || !pages || count == 0)
        return 0;
    
    spin_lock_irqsave(&pool->lock, flags);
    
    /* Check if we have enough */
    if (atomic_long_read(&pool->free) < count) {
        spin_unlock_irqrestore(&pool->lock, flags);
        ps_stats_inc(pool->dev, PS_STAT_ALLOC_FAIL);
        return 0;
    }
    
    /* Allocate all pages */
    for (i = 0; i < count; i++) {
        pages[i] = pool->free_list;
        pool->free_list = pages[i]->zone_device_data;
        pages[i]->zone_device_data = pool->dev;
    }
    
    atomic_long_sub(count, &pool->free);
    
    spin_unlock_irqrestore(&pool->lock, flags);
    
    return count;
}

/*
 * ps_pool_free_bulk - Return multiple pages
 * @pool: Pool to return to
 * @pages: Array of page pointers
 * @count: Number of pages to return
 */
void ps_pool_free_bulk(struct ps_pool *pool,
                       struct page **pages,
                       unsigned long count)
{
    unsigned long flags;
    unsigned long i;
    
    if (!pool || !pages || count == 0)
        return;
    
    spin_lock_irqsave(&pool->lock, flags);
    
    for (i = 0; i < count; i++) {
        if (pages[i]) {
            pages[i]->zone_device_data = pool->free_list;
            pool->free_list = pages[i];
        }
    }
    
    atomic_long_add(count, &pool->free);
    
    spin_unlock_irqrestore(&pool->lock, flags);
}
