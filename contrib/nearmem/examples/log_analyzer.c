/*
 * log_analyzer.c - Near-Memory Computing Demo
 *
 * The Showcase: GPU-accelerated log analysis without data copies.
 *
 * Traditional approach:
 *   1. Load 50 GB logs into RAM          (50 GB disk I/O)
 *   2. Copy to GPU                       (50 GB PCIe, ~4 seconds)
 *   3. GPU grep                          (fast)
 *   4. Copy results back                 (small, but still PCIe)
 *   Total: ~4+ seconds just in transfers
 *
 * Near-Memory approach:
 *   1. Load logs directly to VRAM        (50 GB disk → VRAM via ramdisk)
 *   2. CPU can read/write via mmap       (only touches what it needs)
 *   3. GPU operates in-place             (no transfer!)
 *   4. CPU reads results via mmap        (kilobytes, not gigabytes)
 *   Total: Data never crosses PCIe twice
 *
 * Usage:
 *   # Load logs into pseudoscopic ramdisk
 *   sudo modprobe pseudoscopic mode=ramdisk
 *   sudo dd if=access.log of=/dev/psdisk0 bs=1M
 *   
 *   # Run analyzer
 *   ./log_analyzer /dev/psdisk0 "ERROR"
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>

#include "nearmem.h"

/* Timing helper */
static double get_time_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Format byte size for humans */
static void format_size(size_t bytes, char *buf, size_t buf_len)
{
    const char *units[] = {"B", "KB", "MB", "GB", "TB"};
    int unit = 0;
    double size = bytes;
    
    while (size >= 1024 && unit < 4) {
        size /= 1024;
        unit++;
    }
    
    snprintf(buf, buf_len, "%.2f %s", size, units[unit]);
}

/*
 * Demo 1: Pattern search (grep)
 */
static void demo_grep(nearmem_ctx_t *ctx, nearmem_region_t *logs,
                      const char *pattern)
{
    double start, end;
    int64_t first_match;
    uint64_t match_count;
    size_t pattern_len = strlen(pattern);
    
    printf("\n=== Demo 1: Pattern Search (grep) ===\n");
    printf("Pattern: \"%s\"\n", pattern);
    printf("Data size: %zu bytes\n", logs->size);
    
    /* Find first match */
    start = get_time_ms();
    nearmem_find(ctx, logs, pattern, pattern_len, &first_match);
    end = get_time_ms();
    
    if (first_match >= 0) {
        printf("First match at offset: %ld\n", first_match);
        
        /* Show context around match */
        char context[256];
        size_t ctx_start = (first_match > 40) ? first_match - 40 : 0;
        size_t ctx_len = (first_match + pattern_len + 40 < logs->size) ? 
                         80 + pattern_len : logs->size - ctx_start;
        
        nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
        memcpy(context, (char*)logs->cpu_ptr + ctx_start, ctx_len);
        context[ctx_len] = '\0';
        
        /* Replace newlines for display */
        for (size_t i = 0; i < ctx_len; i++)
            if (context[i] == '\n') context[i] = ' ';
        
        printf("Context: ...%s...\n", context);
    } else {
        printf("Pattern not found\n");
    }
    printf("Time: %.2f ms\n", end - start);
    
    /* Count all matches */
    start = get_time_ms();
    nearmem_count_matches(ctx, logs, pattern, pattern_len, &match_count);
    end = get_time_ms();
    
    printf("Total matches: %lu\n", match_count);
    printf("Count time: %.2f ms\n", end - start);
    
    /* Calculate throughput */
    double throughput = (logs->size / (1024.0 * 1024.0)) / ((end - start) / 1000.0);
    printf("Throughput: %.2f MB/s\n", throughput);
}

/*
 * Demo 2: Histogram analysis
 */
static void demo_histogram(nearmem_ctx_t *ctx, nearmem_region_t *logs)
{
    double start, end;
    uint64_t histogram[256];
    
    printf("\n=== Demo 2: Byte Histogram ===\n");
    
    start = get_time_ms();
    nearmem_histogram(ctx, logs, histogram);
    end = get_time_ms();
    
    /* Find most common bytes */
    uint64_t max_count = 0;
    int max_byte = 0;
    uint64_t total = 0;
    
    for (int i = 0; i < 256; i++) {
        total += histogram[i];
        if (histogram[i] > max_count) {
            max_count = histogram[i];
            max_byte = i;
        }
    }
    
    printf("Total bytes: %lu\n", total);
    printf("Most common byte: 0x%02X ('%c') - %lu occurrences (%.2f%%)\n",
           max_byte, 
           (max_byte >= 32 && max_byte < 127) ? max_byte : '?',
           max_count,
           100.0 * max_count / total);
    
    /* Print printable character distribution */
    printf("\nPrintable character distribution:\n");
    printf("  Lowercase: ");
    uint64_t lower = 0;
    for (int i = 'a'; i <= 'z'; i++) lower += histogram[i];
    printf("%lu (%.2f%%)\n", lower, 100.0 * lower / total);
    
    printf("  Uppercase: ");
    uint64_t upper = 0;
    for (int i = 'A'; i <= 'Z'; i++) upper += histogram[i];
    printf("%lu (%.2f%%)\n", upper, 100.0 * upper / total);
    
    printf("  Digits:    ");
    uint64_t digits = 0;
    for (int i = '0'; i <= '9'; i++) digits += histogram[i];
    printf("%lu (%.2f%%)\n", digits, 100.0 * digits / total);
    
    printf("  Newlines:  %lu (%.2f%%)\n", 
           histogram['\n'], 100.0 * histogram['\n'] / total);
    
    printf("Time: %.2f ms\n", end - start);
    
    double throughput = (logs->size / (1024.0 * 1024.0)) / ((end - start) / 1000.0);
    printf("Throughput: %.2f MB/s\n", throughput);
}

/*
 * Demo 3: In-place transformation
 */
static void demo_transform(nearmem_ctx_t *ctx, nearmem_region_t *region,
                           size_t size)
{
    double start, end;
    
    printf("\n=== Demo 3: In-Place Transform ===\n");
    printf("Operation: Convert to uppercase\n");
    printf("Size: %zu bytes\n", size);
    
    /* Create uppercase LUT */
    uint8_t lut[256];
    for (int i = 0; i < 256; i++) {
        if (i >= 'a' && i <= 'z')
            lut[i] = i - 32;
        else
            lut[i] = i;
    }
    
    /* Show before */
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    char before[64];
    memcpy(before, region->cpu_ptr, 60);
    before[60] = '\0';
    for (int i = 0; i < 60; i++)
        if (before[i] == '\n') before[i] = ' ';
    printf("Before: %s...\n", before);
    
    /* Transform */
    start = get_time_ms();
    nearmem_transform(ctx, region, lut);
    end = get_time_ms();
    
    /* Show after */
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    char after[64];
    memcpy(after, region->cpu_ptr, 60);
    after[60] = '\0';
    for (int i = 0; i < 60; i++)
        if (after[i] == '\n') after[i] = ' ';
    printf("After:  %s...\n", after);
    
    printf("Time: %.2f ms\n", end - start);
    
    double throughput = (size / (1024.0 * 1024.0)) / ((end - start) / 1000.0);
    printf("Throughput: %.2f MB/s\n", throughput);
}

/*
 * Demo 4: The Zero-Copy Advantage
 */
static void demo_zero_copy_comparison(nearmem_ctx_t *ctx, size_t data_size)
{
    printf("\n=== Demo 4: Zero-Copy Analysis ===\n");
    printf("Data size: %zu MB\n", data_size >> 20);
    
    /* Calculate theoretical times */
    double pcie_bandwidth_gbps = 12.0;  /* PCIe Gen3 x16 practical */
    double gpu_bandwidth_gbps = 300.0;  /* Internal GPU bandwidth */
    
    double pcie_time_ms = (data_size / (1024.0 * 1024.0 * 1024.0)) / 
                          pcie_bandwidth_gbps * 1000.0;
    double gpu_time_ms = (data_size / (1024.0 * 1024.0 * 1024.0)) / 
                         gpu_bandwidth_gbps * 1000.0;
    
    printf("\nTraditional approach (copy to GPU, process, copy back):\n");
    printf("  Copy to GPU:     %.2f ms\n", pcie_time_ms);
    printf("  GPU processing:  %.2f ms (at 300 GB/s internal)\n", gpu_time_ms);
    printf("  Copy from GPU:   %.2f ms\n", pcie_time_ms);
    printf("  ----------------------------\n");
    printf("  Total:           %.2f ms\n", 2 * pcie_time_ms + gpu_time_ms);
    
    printf("\nNear-Memory approach (data stays in VRAM):\n");
    printf("  GPU processing:  %.2f ms (at 300 GB/s internal)\n", gpu_time_ms);
    printf("  Read results:    ~0.01 ms (only read what you need)\n");
    printf("  ----------------------------\n");
    printf("  Total:           %.2f ms\n", gpu_time_ms + 0.01);
    
    double speedup = (2 * pcie_time_ms + gpu_time_ms) / (gpu_time_ms + 0.01);
    printf("\nTheoretical speedup: %.1fx\n", speedup);
    printf("PCIe traffic eliminated: %.2f GB (%.0f%%)\n", 
           (2.0 * data_size) / (1024.0 * 1024.0 * 1024.0),
           100.0);
}

/*
 * Demo 5: Memory bandwidth utilization
 */
static void demo_memset_benchmark(nearmem_ctx_t *ctx, nearmem_region_t *region)
{
    double start, end;
    int iterations = 10;
    
    printf("\n=== Demo 5: Memory Bandwidth Benchmark ===\n");
    printf("Operation: memset (GPU-accelerated fill)\n");
    printf("Size: %zu MB\n", region->size >> 20);
    
    /* Warmup */
    nearmem_memset(ctx, region, 0xAA, 0, 0);
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    /* Benchmark */
    start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        nearmem_memset(ctx, region, (uint8_t)i, 0, 0);
    }
    nearmem_sync(ctx, NEARMEM_SYNC_GPU_TO_CPU);
    end = get_time_ms();
    
    double total_bytes = (double)region->size * iterations;
    double total_time_s = (end - start) / 1000.0;
    double bandwidth_gbps = (total_bytes / (1024.0 * 1024.0 * 1024.0)) / total_time_s;
    
    printf("Iterations: %d\n", iterations);
    printf("Total time: %.2f ms\n", end - start);
    printf("Bandwidth: %.2f GB/s\n", bandwidth_gbps);
    printf("\nFor comparison:\n");
    printf("  PCIe Gen3 x16: ~12 GB/s\n");
    printf("  GPU internal:  ~300 GB/s (P100 HBM2)\n");
    printf("  Achieved:      %.2f GB/s\n", bandwidth_gbps);
}

int main(int argc, char *argv[])
{
    nearmem_ctx_t ctx;
    nearmem_region_t logs;
    const char *device_path;
    const char *pattern = "ERROR";
    char size_str[64];
    int err;
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Near-Memory Computing Demo - Log Analyzer                ║\n");
    printf("║     Data doesn't move. Computation comes to the data.        ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    /* Parse arguments */
    if (argc < 2) {
        printf("\nUsage: %s <device> [pattern]\n", argv[0]);
        printf("  device:  /dev/psdisk0 (pseudoscopic ramdisk)\n");
        printf("  pattern: string to search for (default: ERROR)\n");
        printf("\nSetup:\n");
        printf("  sudo modprobe pseudoscopic mode=ramdisk\n");
        printf("  sudo dd if=access.log of=/dev/psdisk0 bs=1M\n");
        printf("  ./%s /dev/psdisk0 ERROR\n", argv[0]);
        return 1;
    }
    
    device_path = argv[1];
    if (argc > 2)
        pattern = argv[2];
    
    /* Initialize near-memory context */
    printf("\nInitializing near-memory context...\n");
    err = nearmem_init(&ctx, device_path, 0);
    if (err != NEARMEM_OK) {
        fprintf(stderr, "Failed to initialize: %s\n", nearmem_strerror(err));
        return 1;
    }
    
    format_size(ctx.ps_size, size_str, sizeof(size_str));
    printf("VRAM capacity: %s\n", size_str);
    
    /* Map entire device as our "logs" region */
    err = nearmem_map_offset(&ctx, &logs, 0, ctx.ps_size);
    if (err != NEARMEM_OK) {
        fprintf(stderr, "Failed to map region: %s\n", nearmem_strerror(err));
        nearmem_shutdown(&ctx);
        return 1;
    }
    
    printf("Mapped region: %p (CPU) / %p (GPU)\n", logs.cpu_ptr, logs.gpu_ptr);
    
    /* Run demos */
    demo_grep(&ctx, &logs, pattern);
    demo_histogram(&ctx, &logs);
    
    /* Only transform a small portion to preserve original data */
    nearmem_region_t small_region;
    nearmem_alloc(&ctx, &small_region, 1024 * 1024);  /* 1 MB */
    nearmem_memcpy(&ctx, &small_region, 0, &logs, 0, 1024 * 1024);
    demo_transform(&ctx, &small_region, small_region.size);
    
    demo_zero_copy_comparison(&ctx, ctx.ps_size);
    demo_memset_benchmark(&ctx, &small_region);
    
    /* Cleanup */
    printf("\n=== Cleanup ===\n");
    nearmem_free(&ctx, &small_region);
    nearmem_shutdown(&ctx);
    printf("Done.\n");
    
    return 0;
}
