#ifndef OMEGA_HASHBELT_H
#define OMEGA_HASHBELT_H
/*
 * omega_hashbelt.h — Lock-Free Hashbelt Allocator
 *
 * A circular array of hash buckets that rotates over time.
 * Old entries age out automatically — "forget the right things."
 *
 * Lock-free insertion via atomic CAS on bucket heads.
 * No ABA problem: entries tagged with belt epoch.
 * Cache-line aligned to prevent false sharing.
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdatomic.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef CACHE_LINE_SIZE
#define CACHE_LINE_SIZE 64
#endif

#define HB_ALIGNED __attribute__((aligned(CACHE_LINE_SIZE)))

/* ── Entry ───────────────────────────────────────────────────── */

typedef struct hb_entry {
    uint64_t                    key;
    uint64_t                    value;
    uint32_t                    belt_epoch;
    _Atomic(struct hb_entry *)  next;
} hb_entry_t;

/* ── Bucket (cache-line padded) ──────────────────────────────── */

typedef struct {
    _Atomic(hb_entry_t *)   head;
    _Atomic(uint32_t)       count;
    uint8_t _pad[CACHE_LINE_SIZE - sizeof(_Atomic(hb_entry_t*)) - sizeof(_Atomic(uint32_t))];
} HB_ALIGNED hb_bucket_t;

/* ── Pool: lock-free slab allocator ──────────────────────────── */

typedef struct {
    hb_entry_t          *slab;
    _Atomic(uint32_t)    alloc_cursor;
    _Atomic(hb_entry_t*) freelist;
    uint32_t             capacity;
} hb_pool_t;

/* ── Belt ────────────────────────────────────────────────────── */

typedef struct {
    hb_bucket_t        *buckets;
    uint32_t            num_buckets;
    uint32_t            bucket_mask;
    hb_pool_t           pool;
    _Atomic(uint32_t)   cursor;
    _Atomic(uint32_t)   epoch;
    uint32_t            window_size;
    _Atomic(uint64_t)   total_inserts;
    _Atomic(uint64_t)   total_lookups;
    _Atomic(uint64_t)   total_hits;
    _Atomic(uint64_t)   total_evictions;
} hb_belt_t;

int         hb_belt_init(hb_belt_t *b, uint32_t nbuckets, uint32_t window, uint32_t pool_cap);
void        hb_belt_destroy(hb_belt_t *b);
hb_entry_t *hb_belt_insert(hb_belt_t *b, uint64_t key, uint64_t value);
uint64_t    hb_belt_lookup(hb_belt_t *b, uint64_t key, bool *found);
uint32_t    hb_belt_advance(hb_belt_t *b);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_HASHBELT_H */

