/* SPDX-License-Identifier: MIT */
/*
 * analytics.c - Near-Memory Data Analytics
 *
 * Demonstrates bulk analytics on large datasets that live in VRAM.
 * Only scalar results (sums, maxes, histograms) cross PCIe.
 *
 * Scenario: 10GB of sensor data (2.5 billion float32 values)
 *   - Compute sum
 *   - Find maximum
 *   - Sort the data
 *   - All without moving 10GB back and forth
 *
 * Traditional: 10GB × 2 × N_ops = 20GB × N_ops PCIe traffic
 * NMC: 10GB load + few bytes per result
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "nmc.h"

#define DATA_SIZE_MB 1024  /* 1GB for demo, scale up for production */
#define NUM_FLOATS (DATA_SIZE_MB * 1024 * 1024 / sizeof(float))

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

int main(int argc, char **argv)
{
    nmc_context_t *ctx = NULL;
    nmc_region_t *data = NULL;
    nmc_error_t err;
    float *host_data = NULL;
    double t_start, elapsed;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║   Near-Memory Data Analytics - Asymmetric Solutions      ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    /*
     * Initialize NMC
     */
    printf("Initializing...\n");
    err = nmc_init(NULL, -1, &ctx);
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "NMC init failed: %s\n", nmc_error_string(err));
        fprintf(stderr, "\nMake sure pseudoscopic is loaded:\n");
        fprintf(stderr, "  sudo modprobe pseudoscopic mode=ramdisk\n");
        return 1;
    }
    
    printf("  VRAM: %zu MB available\n\n", nmc_get_available(ctx) >> 20);
    
    /*
     * Generate test data
     */
    printf("Generating %d MB of test data (%.2f billion floats)...\n",
           DATA_SIZE_MB, NUM_FLOATS / 1e9);
    
    host_data = malloc(NUM_FLOATS * sizeof(float));
    if (!host_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        goto cleanup;
    }
    
    /* Fill with interesting data - sine wave with noise */
    srand(42);
    for (size_t i = 0; i < NUM_FLOATS; i++) {
        float t = (float)i / NUM_FLOATS * 100.0f * M_PI;
        float signal = sinf(t) * 100.0f;
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 10.0f;
        host_data[i] = signal + noise;
    }
    printf("  Data range: approximately [-105, +105]\n");
    printf("  Expected sum: approximately 0 (sine wave)\n\n");
    
    /*
     * Allocate VRAM
     */
    printf("Allocating VRAM region...\n");
    err = nmc_alloc(ctx, NUM_FLOATS * sizeof(float), 
                    NMC_MEM_READ | NMC_MEM_WRITE, &data);
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "Allocation failed: %s\n", nmc_error_string(err));
        goto cleanup;
    }
    printf("  Allocated: %zu MB\n\n", nmc_size(data) >> 20);
    
    /*
     * Load data to VRAM (ONE-TIME transfer)
     */
    printf(">>> Loading data to VRAM (one-time PCIe transfer)...\n");
    t_start = get_time_sec();
    
    err = nmc_load(data, 0, host_data, NUM_FLOATS * sizeof(float));
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "Load failed: %s\n", nmc_error_string(err));
        goto cleanup;
    }
    
    nmc_sync(ctx, NMC_SYNC_CPU_TO_GPU);
    elapsed = get_time_sec() - t_start;
    
    printf("  Loaded %d MB in %.3f sec (%.2f GB/s)\n\n",
           DATA_SIZE_MB, elapsed, DATA_SIZE_MB / 1024.0 / elapsed);
    
    /* Free host memory - we don't need it anymore!
     * This is the key insight: once in VRAM, host memory is free. */
    free(host_data);
    host_data = NULL;
    printf(">>> Host memory freed - all data now lives in VRAM\n\n");
    
    /*
     * Analytics Operations (all in VRAM, only scalars cross PCIe)
     */
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("ANALYTICS (data stays in VRAM, only results cross PCIe)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    /* Operation 1: Sum */
    {
        float sum;
        printf("1. Computing sum of %.2f billion floats...\n", NUM_FLOATS / 1e9);
        t_start = get_time_sec();
        
        err = nmc_reduce_sum_f32(data, 0, NUM_FLOATS, &sum, NULL);
        
        elapsed = get_time_sec() - t_start;
        if (err == NMC_SUCCESS) {
            printf("   Result: %e\n", sum);
            printf("   Time: %.3f sec\n", elapsed);
            printf("   Throughput: %.2f GB/s (GPU internal bandwidth)\n", 
                   DATA_SIZE_MB / 1024.0 / elapsed);
            printf("   PCIe traffic: 4 bytes (just the result!)\n\n");
        } else {
            printf("   Failed: %s\n\n", nmc_error_string(err));
        }
    }
    
    /* Operation 2: Find Maximum */
    {
        float max_val;
        uint64_t max_idx;
        printf("2. Finding maximum value and index...\n");
        t_start = get_time_sec();
        
        err = nmc_reduce_max_f32(data, 0, NUM_FLOATS, &max_val, &max_idx, NULL);
        
        elapsed = get_time_sec() - t_start;
        if (err == NMC_SUCCESS) {
            printf("   Max value: %f at index %lu\n", max_val, max_idx);
            printf("   Time: %.3f sec\n", elapsed);
            printf("   PCIe traffic: 12 bytes (value + index)\n\n");
        } else {
            printf("   Failed: %s\n\n", nmc_error_string(err));
        }
    }
    
    /* Operation 3: Sort */
    {
        printf("3. Sorting %.2f billion floats in-place...\n", NUM_FLOATS / 1e9);
        t_start = get_time_sec();
        
        err = nmc_sort_f32(data, 0, NUM_FLOATS, NULL);
        
        elapsed = get_time_sec() - t_start;
        if (err == NMC_SUCCESS) {
            printf("   Time: %.3f sec\n", elapsed);
            printf("   Throughput: %.2f GB/s\n", DATA_SIZE_MB / 1024.0 / elapsed);
            printf("   PCIe traffic: 0 bytes (entirely in VRAM!)\n\n");
            
            /* Verify by reading first and last elements */
            float first, last;
            nmc_extract(data, 0, &first, sizeof(float));
            nmc_extract(data, (NUM_FLOATS - 1) * sizeof(float), &last, sizeof(float));
            printf("   Verification: first = %f, last = %f\n\n", first, last);
        } else {
            printf("   Failed: %s\n\n", nmc_error_string(err));
        }
    }
    
    /*
     * Summary
     */
    printf("═══════════════════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════\n");
    
    size_t traditional_pcie = (size_t)DATA_SIZE_MB * 1024 * 1024 * 2 * 3;
    size_t nmc_pcie = (size_t)DATA_SIZE_MB * 1024 * 1024 + 16;
    
    printf("\nTraditional approach PCIe traffic:\n");
    printf("  3 operations × %d MB × 2 (to GPU + back) = %zu MB\n",
           DATA_SIZE_MB, traditional_pcie >> 20);
    
    printf("\nNMC approach PCIe traffic:\n");
    printf("  1 load + scalar results = %zu MB + 16 bytes\n",
           (size_t)DATA_SIZE_MB);
    
    printf("\nPCIe reduction: %.1fx less traffic\n\n",
           (double)traditional_pcie / nmc_pcie);
    
    nmc_print_stats(ctx, stdout);

cleanup:
    if (host_data)
        free(host_data);
    if (data)
        nmc_free(data);
    if (ctx)
        nmc_shutdown(ctx);
    
    return 0;
}
