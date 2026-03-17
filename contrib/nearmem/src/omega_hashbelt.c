/*
 * omega_hashbelt.c — Lock-Free Hashbelt Implementation
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_hashbelt.h"
#include <stdlib.h>
#include <string.h>

/* ── Pool: lock-free slab with freelist recycling ────────────── */

static hb_entry_t *pool_alloc(hb_pool_t *p)
{
    /* Try freelist first (lock-free pop via CAS) */
    hb_entry_t *head = atomic_load_explicit(&p->freelist, memory_order_acquire);
    while (head) {
        hb_entry_t *next = atomic_load_explicit(&head->next, memory_order_relaxed);
        if (atomic_compare_exchange_weak_explicit(&p->freelist, &head, next,
                memory_order_release, memory_order_relaxed))
            return head;
    }
    /* Fall through to slab bump allocation */
    uint32_t idx = atomic_fetch_add_explicit(&p->alloc_cursor, 1, memory_order_relaxed);
    if (idx >= p->capacity) return NULL;
    return &p->slab[idx];
}

static void pool_free(hb_pool_t *p, hb_entry_t *entry)
{
    /* Lock-free push to freelist via CAS */
    hb_entry_t *head = atomic_load_explicit(&p->freelist, memory_order_relaxed);
    do {
        atomic_store_explicit(&entry->next, head, memory_order_relaxed);
    } while (!atomic_compare_exchange_weak_explicit(&p->freelist, &head, entry,
                memory_order_release, memory_order_relaxed));
}

/* ── Belt ────────────────────────────────────────────────────── */

int hb_belt_init(hb_belt_t *b, uint32_t nbuckets, uint32_t window, uint32_t pool_cap)
{
    memset(b, 0, sizeof(*b));
    /* Enforce power-of-2 bucket count */
    uint32_t n = 1; while (n < nbuckets) n <<= 1;
    b->buckets = (hb_bucket_t *)calloc(n, sizeof(hb_bucket_t));
    if (!b->buckets) return -1;
    b->num_buckets = n;
    b->bucket_mask = n - 1;
    b->window_size = (window <= n) ? window : n;

    b->pool.slab = (hb_entry_t *)calloc(pool_cap, sizeof(hb_entry_t));
    if (!b->pool.slab) { free(b->buckets); return -1; }
    b->pool.capacity = pool_cap;
    atomic_init(&b->pool.alloc_cursor, 0);
    atomic_init(&b->pool.freelist, NULL);
    return 0;
}

void hb_belt_destroy(hb_belt_t *b)
{
    free(b->pool.slab);
    free(b->buckets);
    memset(b, 0, sizeof(*b));
}

/* Stafford variant 13 finalizer */
static inline uint64_t hb_hash(uint64_t k) {
    k ^= k >> 30; k *= 0xBF58476D1CE4E5B9ULL;
    k ^= k >> 27; k *= 0x94D049BB133111EBULL;
    k ^= k >> 31; return k;
}

hb_entry_t *hb_belt_insert(hb_belt_t *b, uint64_t key, uint64_t value)
{
    hb_entry_t *entry = pool_alloc(&b->pool);
    if (!entry) return NULL;
    entry->key        = key;
    entry->value      = value;
    entry->belt_epoch = atomic_load_explicit(&b->epoch, memory_order_relaxed);

    uint32_t idx = (uint32_t)(hb_hash(key) & b->bucket_mask);
    hb_bucket_t *bucket = &b->buckets[idx];

    /* Lock-free push to bucket chain via CAS */
    hb_entry_t *head = atomic_load_explicit(&bucket->head, memory_order_acquire);
    do {
        atomic_store_explicit(&entry->next, head, memory_order_relaxed);
    } while (!atomic_compare_exchange_weak_explicit(&bucket->head, &head, entry,
                memory_order_release, memory_order_relaxed));

    atomic_fetch_add_explicit(&bucket->count, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&b->total_inserts, 1, memory_order_relaxed);
    return entry;
}

uint64_t hb_belt_lookup(hb_belt_t *b, uint64_t key, bool *found)
{
    atomic_fetch_add_explicit(&b->total_lookups, 1, memory_order_relaxed);
    uint32_t idx = (uint32_t)(hb_hash(key) & b->bucket_mask);
    uint32_t cur_epoch = atomic_load_explicit(&b->epoch, memory_order_acquire);
    uint32_t min_epoch = cur_epoch - b->window_size;

    hb_entry_t *e = atomic_load_explicit(&b->buckets[idx].head, memory_order_acquire);
    while (e) {
        if (e->key == key && e->belt_epoch >= min_epoch) {
            *found = true;
            atomic_fetch_add_explicit(&b->total_hits, 1, memory_order_relaxed);
            return e->value;
        }
        e = atomic_load_explicit(&e->next, memory_order_relaxed);
    }
    *found = false;
    return 0;
}

uint32_t hb_belt_advance(hb_belt_t *b)
{
    uint32_t cur = atomic_fetch_add_explicit(&b->cursor, 1, memory_order_acq_rel);
    uint32_t idx = cur & b->bucket_mask;
    hb_bucket_t *bucket = &b->buckets[idx];

    /* Sweep: atomically detach the entire chain */
    hb_entry_t *chain = atomic_exchange_explicit(&bucket->head, NULL, memory_order_acq_rel);
    uint32_t evicted = 0;
    while (chain) {
        hb_entry_t *next = atomic_load_explicit(&chain->next, memory_order_relaxed);
        pool_free(&b->pool, chain);
        chain = next;
        evicted++;
    }
    atomic_store_explicit(&bucket->count, 0, memory_order_relaxed);
    atomic_fetch_add_explicit(&b->epoch, 1, memory_order_release);
    atomic_fetch_add_explicit(&b->total_evictions, evicted, memory_order_relaxed);
    return evicted;
}

