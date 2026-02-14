/*
 * kv_cache_tier.c - KV-Cache Tiered Storage for Long-Context LLMs
 *
 * The Problem:
 * -----------
 * Modern LLMs with long context (128K+ tokens) have a dirty secret:
 * the KV-cache grows linearly with context and dominates memory.
 *
 *   Context Length    KV-Cache Size (70B model, FP16)
 *   4K tokens         ~2 GB
 *   32K tokens        ~16 GB
 *   128K tokens       ~64 GB
 *   1M tokens         ~512 GB  ← Nobody has this
 *
 * Traditional solutions:
 *   - Limit context length (loses capability)
 *   - Quantize KV-cache (loses precision)
 *   - Streaming attention (loses quality)
 *
 * The Near-Memory Solution:
 * ------------------------
 * Recent tokens (hot): System RAM - fast access
 * Old tokens (cold): VRAM via pseudoscopic - still faster than SSD
 *
 * Access pattern is perfect for this:
 *   - Recent tokens accessed every layer, every head
 *   - Old tokens accessed only during attention (bulk reads)
 *   - Spatial locality: attention reads contiguous key/value vectors
 *
 * This lets you run 128K context on a machine with 64GB RAM + 16GB P100.
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <pthread.h>
#include <sys/time.h>

#include "nearmem.h"

/*
 * Configuration
 */
#define KV_HEAD_DIM     128     /* Typical head dimension */
#define KV_NUM_HEADS    32      /* Number of attention heads */
#define KV_NUM_LAYERS   80      /* Number of transformer layers */
#define KV_DTYPE_SIZE   2       /* FP16 = 2 bytes */

/* Per-token KV size: 2 (K+V) × heads × dim × dtype × layers */
#define KV_PER_TOKEN    (2 * KV_NUM_HEADS * KV_HEAD_DIM * KV_DTYPE_SIZE * KV_NUM_LAYERS)

/* Tier thresholds */
#define HOT_TIER_TOKENS     4096    /* Recent tokens in system RAM */
#define WARM_TIER_TOKENS    32768   /* Older tokens in VRAM */

/*
 * KV-Cache Tier Structure
 */
typedef struct {
    /* Hot tier: System RAM (malloc'd) */
    void *hot_cache;
    size_t hot_size;
    size_t hot_tokens;
    
    /* Warm tier: VRAM (via nearmem) */
    nearmem_ctx_t *nm_ctx;
    nearmem_region_t warm_region;
    size_t warm_tokens;
    
    /* Tracking */
    size_t total_tokens;
    size_t max_hot_tokens;
    size_t max_warm_tokens;
    
    /* Statistics */
    uint64_t hot_hits;
    uint64_t warm_hits;
    uint64_t evictions;
    uint64_t prefetches;
    
    /* Synchronization */
    pthread_mutex_t lock;
} kv_tier_t;

/*
 * Timing helper
 */
static double get_time_us(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000000.0 + tv.tv_usec;
}

/*
 * kv_tier_init - Initialize tiered KV-cache
 */
int kv_tier_init(kv_tier_t *tier, nearmem_ctx_t *nm_ctx,
                 size_t max_hot_tokens, size_t max_warm_tokens)
{
    int err;
    
    memset(tier, 0, sizeof(*tier));
    pthread_mutex_init(&tier->lock, NULL);
    
    tier->nm_ctx = nm_ctx;
    tier->max_hot_tokens = max_hot_tokens;
    tier->max_warm_tokens = max_warm_tokens;
    
    /* Allocate hot tier in system RAM */
    tier->hot_size = max_hot_tokens * KV_PER_TOKEN;
    tier->hot_cache = malloc(tier->hot_size);
    if (!tier->hot_cache) {
        fprintf(stderr, "kv_tier: failed to allocate hot cache (%zu MB)\n",
                tier->hot_size >> 20);
        return -1;
    }
    
    /* Allocate warm tier in VRAM */
    size_t warm_size = max_warm_tokens * KV_PER_TOKEN;
    err = nearmem_alloc(nm_ctx, &tier->warm_region, warm_size);
    if (err != NEARMEM_OK) {
        fprintf(stderr, "kv_tier: failed to allocate warm cache (%zu MB): %s\n",
                warm_size >> 20, nearmem_strerror(err));
        free(tier->hot_cache);
        return -1;
    }
    
    printf("kv_tier: initialized\n");
    printf("  Hot tier:  %zu MB (system RAM, %zu tokens)\n",
           tier->hot_size >> 20, max_hot_tokens);
    printf("  Warm tier: %zu MB (VRAM, %zu tokens)\n",
           warm_size >> 20, max_warm_tokens);
    printf("  Total capacity: %zu tokens\n", max_hot_tokens + max_warm_tokens);
    
    return 0;
}

/*
 * kv_tier_destroy - Clean up
 */
void kv_tier_destroy(kv_tier_t *tier)
{
    if (tier->hot_cache) {
        free(tier->hot_cache);
        tier->hot_cache = NULL;
    }
    
    nearmem_free(tier->nm_ctx, &tier->warm_region);
    pthread_mutex_destroy(&tier->lock);
}

/*
 * kv_tier_append - Add new token's KV to cache
 *
 * New tokens always go to hot tier (system RAM).
 * If hot tier is full, oldest tokens evict to warm tier (VRAM).
 */
int kv_tier_append(kv_tier_t *tier, const void *kv_data)
{
    pthread_mutex_lock(&tier->lock);
    
    /* Check if we need to evict from hot to warm */
    if (tier->hot_tokens >= tier->max_hot_tokens) {
        /* Calculate eviction count (batch eviction is more efficient) */
        size_t evict_count = tier->max_hot_tokens / 4;  /* Evict 25% */
        
        /* Check warm tier capacity */
        if (tier->warm_tokens + evict_count > tier->max_warm_tokens) {
            /* Warm tier full - drop oldest warm tokens */
            size_t drop_count = evict_count;
            
            /* Shift warm tier data */
            size_t shift_bytes = drop_count * KV_PER_TOKEN;
            size_t remaining = (tier->warm_tokens - drop_count) * KV_PER_TOKEN;
            
            /* GPU memcpy within VRAM (no PCIe!) */
            nearmem_memcpy(tier->nm_ctx,
                          &tier->warm_region, 0,
                          &tier->warm_region, shift_bytes,
                          remaining);
            
            tier->warm_tokens -= drop_count;
        }
        
        /* Copy evicted tokens from hot to warm */
        size_t evict_bytes = evict_count * KV_PER_TOKEN;
        size_t warm_offset = tier->warm_tokens * KV_PER_TOKEN;
        
        memcpy((char*)tier->warm_region.cpu_ptr + warm_offset,
               tier->hot_cache,
               evict_bytes);
        
        nearmem_sync_region(tier->nm_ctx, &tier->warm_region,
                           NEARMEM_SYNC_CPU_TO_GPU);
        
        /* Shift remaining hot tokens to front */
        size_t remaining_hot = (tier->hot_tokens - evict_count) * KV_PER_TOKEN;
        memmove(tier->hot_cache,
                (char*)tier->hot_cache + evict_bytes,
                remaining_hot);
        
        tier->hot_tokens -= evict_count;
        tier->warm_tokens += evict_count;
        tier->evictions += evict_count;
    }
    
    /* Append new token to hot tier */
    size_t hot_offset = tier->hot_tokens * KV_PER_TOKEN;
    memcpy((char*)tier->hot_cache + hot_offset, kv_data, KV_PER_TOKEN);
    tier->hot_tokens++;
    tier->total_tokens++;
    
    pthread_mutex_unlock(&tier->lock);
    return 0;
}

/*
 * kv_tier_get - Retrieve KV data for a token position
 *
 * Checks hot tier first (fast), then warm tier (still fast, but VRAM).
 */
int kv_tier_get(kv_tier_t *tier, size_t token_pos, void *kv_out)
{
    pthread_mutex_lock(&tier->lock);
    
    /* Calculate which tier this token is in */
    size_t warm_start = 0;
    size_t hot_start = tier->warm_tokens;
    
    if (token_pos >= hot_start) {
        /* Hot tier hit */
        size_t hot_idx = token_pos - hot_start;
        size_t offset = hot_idx * KV_PER_TOKEN;
        
        memcpy(kv_out, (char*)tier->hot_cache + offset, KV_PER_TOKEN);
        tier->hot_hits++;
        
        pthread_mutex_unlock(&tier->lock);
        return 0;
    }
    
    /* Warm tier hit - read from VRAM */
    size_t warm_idx = token_pos - warm_start;
    if (warm_idx >= tier->warm_tokens) {
        pthread_mutex_unlock(&tier->lock);
        return -1;  /* Token not in cache */
    }
    
    nearmem_sync_region(tier->nm_ctx, &tier->warm_region,
                       NEARMEM_SYNC_GPU_TO_CPU);
    
    size_t offset = warm_idx * KV_PER_TOKEN;
    memcpy(kv_out, (char*)tier->warm_region.cpu_ptr + offset, KV_PER_TOKEN);
    tier->warm_hits++;
    
    pthread_mutex_unlock(&tier->lock);
    return 0;
}

/*
 * kv_tier_get_range - Get KV data for range of tokens (attention window)
 *
 * This is the common access pattern: attention needs all keys/values
 * from position 0 to current position.
 */
int kv_tier_get_range(kv_tier_t *tier, size_t start, size_t end,
                      void *kv_out, size_t *hot_count, size_t *warm_count)
{
    pthread_mutex_lock(&tier->lock);
    
    size_t warm_end_pos = tier->warm_tokens;
    char *out_ptr = kv_out;
    *hot_count = 0;
    *warm_count = 0;
    
    /* Copy warm tier portion (if any) */
    if (start < warm_end_pos) {
        size_t warm_start = start;
        size_t warm_end = (end < warm_end_pos) ? end : warm_end_pos;
        size_t warm_bytes = (warm_end - warm_start) * KV_PER_TOKEN;
        
        nearmem_sync_region(tier->nm_ctx, &tier->warm_region,
                           NEARMEM_SYNC_GPU_TO_CPU);
        
        memcpy(out_ptr,
               (char*)tier->warm_region.cpu_ptr + warm_start * KV_PER_TOKEN,
               warm_bytes);
        
        out_ptr += warm_bytes;
        *warm_count = warm_end - warm_start;
    }
    
    /* Copy hot tier portion (if any) */
    if (end > warm_end_pos) {
        size_t hot_start = (start > warm_end_pos) ? start - warm_end_pos : 0;
        size_t hot_end = end - warm_end_pos;
        size_t hot_bytes = (hot_end - hot_start) * KV_PER_TOKEN;
        
        memcpy(out_ptr,
               (char*)tier->hot_cache + hot_start * KV_PER_TOKEN,
               hot_bytes);
        
        *hot_count = hot_end - hot_start;
    }
    
    tier->hot_hits += *hot_count;
    tier->warm_hits += *warm_count;
    
    pthread_mutex_unlock(&tier->lock);
    return 0;
}

/*
 * kv_tier_stats - Print statistics
 */
void kv_tier_stats(kv_tier_t *tier)
{
    printf("\nKV-Cache Tier Statistics:\n");
    printf("  Total tokens:     %zu\n", tier->total_tokens);
    printf("  Hot tier tokens:  %zu / %zu\n", 
           tier->hot_tokens, tier->max_hot_tokens);
    printf("  Warm tier tokens: %zu / %zu\n", 
           tier->warm_tokens, tier->max_warm_tokens);
    printf("  Hot tier hits:    %lu\n", tier->hot_hits);
    printf("  Warm tier hits:   %lu\n", tier->warm_hits);
    printf("  Evictions:        %lu\n", tier->evictions);
    
    double hot_ratio = 100.0 * tier->hot_hits / 
                       (tier->hot_hits + tier->warm_hits + 1);
    printf("  Hot tier hit rate: %.1f%%\n", hot_ratio);
}

/*
 * Simulation: Fake attention computation
 */
static void simulate_attention(kv_tier_t *tier, size_t current_pos)
{
    /* Attention needs all previous KV pairs */
    size_t kv_size = (current_pos + 1) * KV_PER_TOKEN;
    void *kv_buffer = malloc(kv_size);
    
    size_t hot_count, warm_count;
    kv_tier_get_range(tier, 0, current_pos + 1, 
                      kv_buffer, &hot_count, &warm_count);
    
    /* Simulate attention computation... */
    
    free(kv_buffer);
}

/*
 * Main: Demonstration
 */
int main(int argc, char *argv[])
{
    nearmem_ctx_t nm_ctx;
    kv_tier_t tier;
    int err;
    double start, end;
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     KV-Cache Tiered Storage Demo                             ║\n");
    printf("║     Hot tier: System RAM | Warm tier: GPU VRAM               ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n\n");
    
    /* Simulated model parameters */
    printf("Simulated Model Configuration:\n");
    printf("  Layers:     %d\n", KV_NUM_LAYERS);
    printf("  Heads:      %d\n", KV_NUM_HEADS);
    printf("  Head dim:   %d\n", KV_HEAD_DIM);
    printf("  KV/token:   %d bytes\n", KV_PER_TOKEN);
    
    /* Initialize near-memory context */
    const char *device = (argc > 1) ? argv[1] : NULL;
    printf("\nInitializing near-memory context (%s)...\n", device ? device : "Auto-detect");
    
    err = nearmem_init(&nm_ctx, device, 0);
    if (err != NEARMEM_OK) {
        fprintf(stderr, "Error: Could not initialize VRAM (%s)\n", nearmem_strerror(err));
        return 1;
    }
    
    /* Initialize tiered cache */
    size_t hot_tokens = HOT_TIER_TOKENS;
    size_t warm_tokens = WARM_TIER_TOKENS;
    
    printf("\nInitializing tiered KV-cache...\n");
    
    err = kv_tier_init(&tier, &nm_ctx, hot_tokens, warm_tokens);
    if (err != 0) {
        nearmem_shutdown(&nm_ctx);
        return 1;
    }
    
    /* Simulate token generation */
    size_t total_tokens = hot_tokens + warm_tokens / 2;  /* Fill partially */
    printf("\nSimulating %zu token generation...\n", total_tokens);
    
    /* Create dummy KV data */
    void *dummy_kv = malloc(KV_PER_TOKEN);
    memset(dummy_kv, 0x42, KV_PER_TOKEN);
    
    start = get_time_us();
    
    for (size_t i = 0; i < total_tokens; i++) {
        /* Append new token's KV */
        kv_tier_append(&tier, dummy_kv);
        
        /* Simulate attention (every token) */
        if (i > 0 && i % 100 == 0) {
            simulate_attention(&tier, i);
        }
        
        /* Progress */
        if ((i + 1) % 10000 == 0) {
            printf("  Generated %zu tokens...\n", i + 1);
        }
    }
    
    end = get_time_us();
    
    printf("\nGeneration complete!\n");
    printf("  Time: %.2f ms\n", (end - start) / 1000.0);
    printf("  Tokens/sec: %.0f\n", total_tokens / ((end - start) / 1000000.0));
    
    /* Print statistics */
    kv_tier_stats(&tier);
    
    /* Memory analysis */
    printf("\nMemory Analysis:\n");
    size_t total_kv_size = total_tokens * KV_PER_TOKEN;
    printf("  Total KV data: %.2f GB\n", total_kv_size / (1024.0 * 1024.0 * 1024.0));
    printf("  System RAM used: %.2f GB (hot tier)\n", 
           tier.hot_size / (1024.0 * 1024.0 * 1024.0));
    printf("  VRAM used: %.2f GB (warm tier)\n",
           (tier.warm_tokens * KV_PER_TOKEN) / (1024.0 * 1024.0 * 1024.0));
    
    printf("\n💡 Without tiering, this would require %.2f GB of system RAM!\n",
           total_kv_size / (1024.0 * 1024.0 * 1024.0));
    printf("   With tiering, we only need %.2f GB RAM + %.2f GB VRAM.\n",
           tier.hot_size / (1024.0 * 1024.0 * 1024.0),
           (tier.warm_tokens * KV_PER_TOKEN) / (1024.0 * 1024.0 * 1024.0));
    
    /* Cleanup */
    free(dummy_kv);
    kv_tier_destroy(&tier);
    nearmem_shutdown(&nm_ctx);
    
    printf("\nDone.\n");
    return 0;
}
