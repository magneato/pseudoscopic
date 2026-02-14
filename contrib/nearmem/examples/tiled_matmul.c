/*
 * tiled_matmul.c - Tiled Matrix Multiplication Demo
 *
 * THE canonical example for demonstrating tiled computation.
 *
 * The Classic Problem:
 * -------------------
 * Matrix multiplication C = A × B requires O(n³) operations
 * but only O(n²) data. The bottleneck is memory bandwidth.
 *
 * For matrices that don't fit in GPU memory:
 *   Traditional: Copy A and B to GPU, compute, copy C back
 *   → 3 × n² transfers
 *
 * With near-memory tiling:
 *   - A and B live in VRAM (via pseudoscopic)
 *   - Load tiles of A and B into shared memory
 *   - Compute partial products
 *   - Accumulate into C
 *   - Only C tiles cross back (or stay in VRAM!)
 *
 * The Double-Buffer Dance:
 * -----------------------
 *   While computing on tile[i]:
 *     - Prefetch tile[i+1] from VRAM to buffer B
 *   When compute finishes:
 *     - Swap buffers (A ↔ B)
 *     - Start compute on new tile
 *     - Prefetch next tile
 *
 *   Result: Memory latency hidden behind compute!
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>

#include "nearmem.h"
#include "nearmem_tile.h"

/* Timing helper */
static double get_time_ms(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 * CPU reference implementation (for verification)
 */
static void matmul_cpu_naive(const float *A, const float *B, float *C,
                              size_t M, size_t N, size_t K)
{
    /* C[M×N] = A[M×K] × B[K×N] */
    for (size_t i = 0; i < M; i++) {
        for (size_t j = 0; j < N; j++) {
            float sum = 0;
            for (size_t k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/*
 * CPU tiled implementation (cache-blocked)
 *
 * This mirrors the GPU tiled approach:
 *   - Process in TILE_SIZE × TILE_SIZE blocks
 *   - Accumulate partial products
 *   - Better cache utilization
 */
static void matmul_cpu_tiled(const float *A, const float *B, float *C,
                              size_t M, size_t N, size_t K,
                              size_t tile_size)
{
    /* Initialize C to zero */
    memset(C, 0, M * N * sizeof(float));
    
    /* Iterate over tiles */
    for (size_t i0 = 0; i0 < M; i0 += tile_size) {
        for (size_t j0 = 0; j0 < N; j0 += tile_size) {
            for (size_t k0 = 0; k0 < K; k0 += tile_size) {
                
                /* Compute tile bounds */
                size_t i_end = (i0 + tile_size < M) ? i0 + tile_size : M;
                size_t j_end = (j0 + tile_size < N) ? j0 + tile_size : N;
                size_t k_end = (k0 + tile_size < K) ? k0 + tile_size : K;
                
                /* Multiply tile A[i0:i_end, k0:k_end] × B[k0:k_end, j0:j_end]
                   and accumulate into C[i0:i_end, j0:j_end] */
                for (size_t i = i0; i < i_end; i++) {
                    for (size_t j = j0; j < j_end; j++) {
                        float sum = C[i * N + j];
                        for (size_t k = k0; k < k_end; k++) {
                            sum += A[i * K + k] * B[k * N + j];
                        }
                        C[i * N + j] = sum;
                    }
                }
            }
        }
    }
}

/*
 * Near-memory tiled implementation
 *
 * Uses the tile iterator API for prefetching and double-buffering.
 */
static void matmul_nearmem_tiled(nearmem_ctx_t *ctx,
                                  nearmem_region_t *A_region,
                                  nearmem_region_t *B_region,
                                  nearmem_region_t *C_region,
                                  size_t M, size_t N, size_t K,
                                  size_t tile_size)
{
    nm_tile_2d_t A_tile, B_tile, C_tile;
    nm_tile_iterator_t A_iter, C_iter;
    double start, end;
    
    printf("\n--- Near-Memory Tiled Matrix Multiply ---\n");
    printf("Tile size: %zu × %zu\n", tile_size, tile_size);
    
    /* 
     * Initialize tiles
     *
     * For C = A × B where A is M×K, B is K×N, C is M×N:
     *   - Tile A row-wise (iterate over M dimension)
     *   - Tile B column-wise (iterate over N dimension)
     *   - Tile C for output
     *
     * For each C tile, we need full rows of A and full columns of B.
     * But we can break the K dimension into chunks (partial products).
     */
    
    /* Tile A as M×K with tiles of tile_size×K (full K per tile) */
    nm_tile_2d_init(&A_tile, ctx, A_region, K, M, 0, K, tile_size, NM_DTYPE_F32);
    nm_tile_2d_set_mode(&A_tile, NM_TILE_READ);
    nm_tile_2d_set_access(&A_tile, NM_ACCESS_SEQUENTIAL);
    
    /* Tile C as M×N with tiles of tile_size×N */
    nm_tile_2d_init(&C_tile, ctx, C_region, N, M, 0, N, tile_size, NM_DTYPE_F32);
    nm_tile_2d_set_mode(&C_tile, NM_TILE_READWRITE);
    
    printf("A tiles: %zu (rows of %zu)\n", A_tile.num_tiles, tile_size);
    printf("C tiles: %zu (rows of %zu)\n", C_tile.num_tiles, tile_size);
    
    /* Zero C first */
    nearmem_memset(ctx, C_region, 0, 0, 0);
    
    start = get_time_ms();
    
    /* 
     * The outer loop processes C in row-strips.
     * For each row-strip of C, we need the corresponding row-strip of A
     * and ALL of B.
     */
    nm_tile_2d_begin(&A_iter, &A_tile);
    nm_tile_2d_begin(&C_iter, &C_tile);
    
    size_t row_tiles = 0;
    
    do {
        /* Get A row-strip (prefetched) */
        float *A_data = nm_tile_get_cpu(&A_iter);
        float *C_data = nm_tile_get_cpu(&C_iter);
        float *B_data = B_region->cpu_ptr;
        
        /* Sync to ensure data is visible */
        nm_tile_sync_gpu(&A_iter);
        nm_tile_sync_gpu(&C_iter);
        
        size_t rows_in_tile = A_iter.current_tile_h;
        
        /* 
         * Compute this strip: C_strip = A_strip × B
         *
         * A_strip is rows_in_tile × K
         * B is K × N
         * C_strip is rows_in_tile × N
         */
        for (size_t i = 0; i < rows_in_tile; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0;
                for (size_t k = 0; k < K; k++) {
                    sum += A_data[i * K + k] * B_data[k * N + j];
                }
                C_data[i * N + j] = sum;
            }
        }
        
        nm_tile_mark_dirty(&C_iter);
        nm_tile_sync_cpu(&C_iter);
        
        row_tiles++;
        
    } while (nm_tile_next(&A_iter) && nm_tile_next(&C_iter));
    
    nm_tile_end(&A_iter);
    nm_tile_end(&C_iter);
    
    end = get_time_ms();
    
    printf("Processed %zu row tiles in %.2f ms\n", row_tiles, end - start);
    
    /* Calculate GFLOPS */
    double gflops = (2.0 * M * N * K) / ((end - start) / 1000.0) / 1e9;
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    nm_tile_2d_destroy(&A_tile);
    nm_tile_2d_destroy(&C_tile);
}

/*
 * Verify results
 */
static double max_error(const float *A, const float *B, size_t n)
{
    double max_err = 0;
    for (size_t i = 0; i < n; i++) {
        double err = fabs(A[i] - B[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

/*
 * Main
 */
int main(int argc, char *argv[])
{
    nearmem_ctx_t ctx;
    nearmem_region_t A_region, B_region, C_region;
    int err;
    double start, end;
    double tile_err = 0;
    double nearmem_err = 0;
    
    /* Default sizes */
    size_t M = 1024;  /* Rows of A and C */
    size_t N = 1024;  /* Cols of B and C */
    size_t K = 1024;  /* Cols of A, Rows of B */
    size_t tile_size = 64;
    
    printf("╔══════════════════════════════════════════════════════════════╗\n");
    printf("║     Tiled Matrix Multiplication Demo                         ║\n");
    printf("║     C[M×N] = A[M×K] × B[K×N]                                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════╝\n");
    
    /* Parse arguments */
    if (argc > 1) M = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) K = atoi(argv[3]);
    if (argc > 4) tile_size = atoi(argv[4]);
    
    printf("\nMatrix dimensions:\n");
    printf("  A: %zu × %zu (%.2f MB)\n", M, K, M * K * sizeof(float) / 1e6);
    printf("  B: %zu × %zu (%.2f MB)\n", K, N, K * N * sizeof(float) / 1e6);
    printf("  C: %zu × %zu (%.2f MB)\n", M, N, M * N * sizeof(float) / 1e6);
    printf("  Tile size: %zu\n", tile_size);
    
    size_t total_bytes = (M * K + K * N + M * N) * sizeof(float);
    printf("  Total memory: %.2f MB\n", total_bytes / 1e6);
    
    /* Allocate matrices in system RAM */
    float *A = malloc(M * K * sizeof(float));
    float *B = malloc(K * N * sizeof(float));
    float *C_ref = malloc(M * N * sizeof(float));
    float *C_tiled = malloc(M * N * sizeof(float));
    float *C_nearmem = malloc(M * N * sizeof(float));
    
    if (!A || !B || !C_ref || !C_tiled || !C_nearmem) {
        fprintf(stderr, "Failed to allocate matrices\n");
        return 1;
    }
    
    /* Initialize with random data */
    printf("\nInitializing matrices...\n");
    srand(42);
    for (size_t i = 0; i < M * K; i++) A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < K * N; i++) B[i] = (float)rand() / RAND_MAX;
    
    /* Reference implementation */
    printf("\n--- CPU Naive Implementation ---\n");
    start = get_time_ms();
    matmul_cpu_naive(A, B, C_ref, M, N, K);
    end = get_time_ms();
    printf("Time: %.2f ms\n", end - start);
    printf("GFLOPS: %.2f\n", (2.0 * M * N * K) / ((end - start) / 1000.0) / 1e9);
    
    /* Tiled CPU implementation */
    printf("\n--- CPU Tiled Implementation ---\n");
    start = get_time_ms();
    matmul_cpu_tiled(A, B, C_tiled, M, N, K, tile_size);
    end = get_time_ms();
    printf("Time: %.2f ms\n", end - start);
    printf("GFLOPS: %.2f\n", (2.0 * M * N * K) / ((end - start) / 1000.0) / 1e9);
    
    /* Verify tiled matches reference */
    tile_err = max_error(C_ref, C_tiled, M * N);
    printf("Max error vs reference: %e\n", tile_err);
    
    /* Near-memory implementation */
    const char *device = (argc > 5) ? argv[5] : NULL;
    
    err = nearmem_init(&ctx, device, 0);
    if (err != NEARMEM_OK) {
        fprintf(stderr, "Error: Near-memory not available (%s)\n", nearmem_strerror(err));
        return 1;
    }

    /* Allocate VRAM regions */
    nearmem_alloc(&ctx, &A_region, M * K * sizeof(float));
    nearmem_alloc(&ctx, &B_region, K * N * sizeof(float));
    nearmem_alloc(&ctx, &C_region, M * N * sizeof(float));
    
    /* Copy A and B to VRAM */
    printf("\nCopying matrices to VRAM...\n");
    memcpy(A_region.cpu_ptr, A, M * K * sizeof(float));
    memcpy(B_region.cpu_ptr, B, K * N * sizeof(float));
    nearmem_sync(&ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    /* Run near-memory tiled multiply */
    matmul_nearmem_tiled(&ctx, &A_region, &B_region, &C_region,
                        M, N, K, tile_size);
    
    /* Copy result back */
    nearmem_sync(&ctx, NEARMEM_SYNC_GPU_TO_CPU);
    memcpy(C_nearmem, C_region.cpu_ptr, M * N * sizeof(float));
    
    /* Verify */
    nearmem_err = max_error(C_ref, C_nearmem, M * N);
    printf("Max error vs reference: %e\n", nearmem_err);
    
    /* Cleanup */
    nearmem_free(&ctx, &A_region);
    nearmem_free(&ctx, &B_region);
    nearmem_free(&ctx, &C_region);
    nearmem_shutdown(&ctx);
    
    /* Summary */
    printf("\n=== Summary ===\n");
    printf("Matrix multiply is compute-bound (O(n³) ops, O(n²) data).\n");
    printf("But large matrices don't fit in GPU memory!\n");
    printf("\nWith near-memory tiling:\n");
    printf("  1. Matrices live in VRAM (16-80 GB on Tesla cards)\n");
    printf("  2. Tiles are prefetched with double-buffering\n");
    printf("  3. GPU computes while next tile loads\n");
    printf("  4. Result stays in VRAM (no copy back needed!)\n");
    printf("\nFor production use:\n");
    printf("  - Replace CPU loops with CUDA kernels\n");
    printf("  - Use shared memory for tile multiplication\n");
    printf("  - Achieved memory bandwidth: internal GPU (~700 GB/s)\n");
    printf("  - Not limited by PCIe (~12 GB/s)\n");
    
    /* Cleanup */
    free(A);
    free(B);
    free(C_ref);
    free(C_tiled);
    free(C_nearmem);
    
    printf("\nDone.\n");
    return 0;
}
