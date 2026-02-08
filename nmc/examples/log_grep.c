/* SPDX-License-Identifier: MIT */
/*
 * log_grep.c - Near-Memory Log Search
 *
 * Demonstrates the NMC pattern: 50GB of logs loaded ONCE,
 * searched multiple times with different patterns.
 * Only matching line offsets cross PCIe.
 *
 * Traditional approach:
 *   Load 50GB to RAM (50 GB PCIe)
 *   For each search:
 *     Copy to GPU (50 GB PCIe)
 *     Search
 *     Copy results back (N KB PCIe)
 *   Total for 10 searches: 550 GB PCIe
 *
 * NMC approach:
 *   Load 50GB to VRAM (50 GB PCIe)
 *   For each search:
 *     GPU searches in-place
 *     Copy results back (N KB PCIe)
 *   Total for 10 searches: 50 GB + 10*N KB PCIe
 *
 * For N = 10KB results: 550 GB vs 50.1 GB = 11x less PCIe traffic
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>

#include "nmc.h"

#define MAX_RESULTS (1024 * 1024)  /* 1M matches max */

static double get_time_sec(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
}

static void print_usage(const char *prog)
{
    fprintf(stderr, "Usage: %s <log_file> <pattern1> [pattern2] ...\n", prog);
    fprintf(stderr, "\n");
    fprintf(stderr, "Loads log file into GPU VRAM, then searches for each pattern.\n");
    fprintf(stderr, "Data crosses PCIe ONCE. Searches happen at GPU memory bandwidth.\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Example:\n");
    fprintf(stderr, "  %s /var/log/syslog \"error\" \"warning\" \"failed\"\n", prog);
}

int main(int argc, char **argv)
{
    nmc_context_t *ctx = NULL;
    nmc_region_t *data_region = NULL;
    nmc_region_t *results_region = NULL;
    nmc_error_t err;
    int fd;
    struct stat st;
    void *file_data;
    double t_start, t_load, t_search;
    size_t total_pcie_saved = 0;
    
    if (argc < 3) {
        print_usage(argv[0]);
        return 1;
    }
    
    const char *log_file = argv[1];
    int num_patterns = argc - 2;
    
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║     Near-Memory Log Search - Asymmetric Solutions        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");
    
    /*
     * Step 1: Initialize NMC
     */
    printf("Initializing NMC...\n");
    err = nmc_init(NULL, -1, &ctx);
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "Failed to initialize NMC: %s\n", nmc_error_string(err));
        fprintf(stderr, "Make sure pseudoscopic is loaded in ramdisk mode:\n");
        fprintf(stderr, "  sudo modprobe pseudoscopic mode=ramdisk\n");
        return 1;
    }
    
    printf("  VRAM capacity: %zu MB\n", nmc_get_capacity(ctx) >> 20);
    printf("  VRAM available: %zu MB\n\n", nmc_get_available(ctx) >> 20);
    
    /*
     * Step 2: Open and mmap log file
     */
    printf("Opening log file: %s\n", log_file);
    fd = open(log_file, O_RDONLY);
    if (fd < 0) {
        perror("open");
        nmc_shutdown(ctx);
        return 1;
    }
    
    if (fstat(fd, &st) < 0) {
        perror("fstat");
        close(fd);
        nmc_shutdown(ctx);
        return 1;
    }
    
    printf("  File size: %zu MB\n", (size_t)st.st_size >> 20);
    
    if ((size_t)st.st_size > nmc_get_available(ctx)) {
        fprintf(stderr, "File too large for available VRAM\n");
        close(fd);
        nmc_shutdown(ctx);
        return 1;
    }
    
    /* mmap the file for efficient loading */
    file_data = mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        nmc_shutdown(ctx);
        return 1;
    }
    
    /*
     * Step 3: Allocate VRAM regions
     */
    printf("\nAllocating VRAM regions...\n");
    
    err = nmc_alloc(ctx, st.st_size, NMC_MEM_READ | NMC_MEM_GPU_ONLY, &data_region);
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "Failed to allocate data region: %s\n", nmc_error_string(err));
        goto cleanup;
    }
    printf("  Data region: %zu MB\n", nmc_size(data_region) >> 20);
    
    err = nmc_alloc(ctx, MAX_RESULTS * sizeof(uint64_t), 
                    NMC_MEM_READ | NMC_MEM_WRITE, &results_region);
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "Failed to allocate results region: %s\n", nmc_error_string(err));
        goto cleanup;
    }
    printf("  Results region: %zu KB\n", nmc_size(results_region) >> 10);
    
    /*
     * Step 4: Load data to VRAM
     * THIS IS THE ONLY TIME THE FULL DATA CROSSES PCIE
     */
    printf("\n>>> Loading data to VRAM (ONE-TIME PCIe transfer)...\n");
    t_start = get_time_sec();
    
    err = nmc_load(data_region, 0, file_data, st.st_size);
    if (err != NMC_SUCCESS) {
        fprintf(stderr, "Failed to load data: %s\n", nmc_error_string(err));
        goto cleanup;
    }
    
    t_load = get_time_sec() - t_start;
    printf("  Loaded %zu MB in %.2f seconds (%.2f GB/s)\n",
           (size_t)st.st_size >> 20, t_load, 
           (st.st_size / 1e9) / t_load);
    
    /* Sync to ensure load complete before searches */
    nmc_sync(ctx, NMC_SYNC_CPU_TO_GPU);
    
    /*
     * Step 5: Search for each pattern
     * Data STAYS IN VRAM - only results cross PCIe
     */
    printf("\n>>> Searching (data stays in VRAM)...\n\n");
    
    for (int i = 0; i < num_patterns; i++) {
        const char *pattern = argv[2 + i];
        uint64_t match_count = 0;
        
        printf("Pattern %d: \"%s\"\n", i + 1, pattern);
        t_start = get_time_sec();
        
        err = nmc_search(data_region, 0, st.st_size,
                        pattern, strlen(pattern),
                        results_region, MAX_RESULTS,
                        &match_count, NULL);
        
        if (err != NMC_SUCCESS) {
            fprintf(stderr, "  Search failed: %s\n", nmc_error_string(err));
            continue;
        }
        
        t_search = get_time_sec() - t_start;
        
        printf("  Found: %lu matches\n", match_count);
        printf("  Time: %.3f seconds\n", t_search);
        printf("  Throughput: %.2f GB/s (internal GPU bandwidth)\n",
               (st.st_size / 1e9) / t_search);
        
        /* Calculate PCIe savings */
        size_t traditional_pcie = st.st_size;  /* Would copy entire file */
        size_t nmc_pcie = match_count * sizeof(uint64_t);  /* Only results */
        printf("  PCIe saved: %.2f MB (traditional would transfer %.2f MB)\n",
               (traditional_pcie - nmc_pcie) / 1e6, traditional_pcie / 1e6);
        
        total_pcie_saved += traditional_pcie - nmc_pcie;
        
        /* Optionally print first few matches */
        if (match_count > 0 && match_count <= 10) {
            uint64_t offsets[10];
            err = nmc_extract(results_region, 0, offsets, 
                             match_count * sizeof(uint64_t));
            if (err == NMC_SUCCESS) {
                printf("  Match offsets: ");
                for (uint64_t j = 0; j < match_count; j++) {
                    printf("%lu ", offsets[j]);
                }
                printf("\n");
            }
        }
        
        printf("\n");
    }
    
    /*
     * Summary
     */
    printf("═══════════════════════════════════════════════════════════\n");
    printf("SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  Patterns searched: %d\n", num_patterns);
    printf("  Data size: %.2f MB\n", st.st_size / 1e6);
    printf("  Total PCIe saved: %.2f MB\n", total_pcie_saved / 1e6);
    printf("  PCIe efficiency: %.1f%% reduction vs traditional\n",
           100.0 * total_pcie_saved / (st.st_size * num_patterns));
    
    nmc_print_stats(ctx, stdout);

cleanup:
    if (file_data && file_data != MAP_FAILED)
        munmap(file_data, st.st_size);
    if (fd >= 0)
        close(fd);
    if (results_region)
        nmc_free(results_region);
    if (data_region)
        nmc_free(data_region);
    if (ctx)
        nmc_shutdown(ctx);
    
    return 0;
}
