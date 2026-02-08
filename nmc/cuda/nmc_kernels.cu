/* SPDX-License-Identifier: MIT */
/*
 * nmc_kernels.cu - GPU Kernels for Near-Memory Computing
 *
 * These kernels operate on data IN VRAM. The key insight:
 * data never crosses PCIe during these operations.
 *
 * Internal GPU bandwidth: ~700 GB/s (HBM2)
 * PCIe bandwidth:         ~12 GB/s
 * Ratio:                  58x
 *
 * Every operation we do here instead of on CPU saves 58x
 * in effective bandwidth.
 *
 * Copyright (C) 2026 Neural Splines LLC
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cub/cub.cuh>  /* NVIDIA CUB for optimized primitives */

#include "nmc.h"

/* Block sizes tuned for compute capability 6.0+ (Pascal) */
#define BLOCK_SIZE 256
#define WARP_SIZE 32

/*
 * ============================================================
 * Transform Kernels
 * ============================================================
 */

/*
 * Byte-wise lookup table transform
 *
 * Each thread processes 4 bytes using shared memory LUT.
 * Achieves near memory-bandwidth-limited throughput.
 */
__global__ void kernel_transform_u8(uint8_t *data, 
                                    size_t size,
                                    const uint8_t *table)
{
    __shared__ uint8_t shared_table[256];
    
    /* Collaboratively load LUT to shared memory */
    if (threadIdx.x < 256) {
        shared_table[threadIdx.x] = table[threadIdx.x];
    }
    __syncthreads();
    
    /* Process 4 bytes per thread for efficiency */
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        /* Load 4 bytes */
        uint32_t val = *((uint32_t *)(data + idx));
        
        /* Transform each byte */
        uint8_t b0 = shared_table[(val >> 0) & 0xFF];
        uint8_t b1 = shared_table[(val >> 8) & 0xFF];
        uint8_t b2 = shared_table[(val >> 16) & 0xFF];
        uint8_t b3 = shared_table[(val >> 24) & 0xFF];
        
        /* Store transformed bytes */
        *((uint32_t *)(data + idx)) = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24);
    }
    else if (idx < size) {
        /* Handle tail */
        for (size_t i = idx; i < size && i < idx + 4; i++) {
            data[i] = shared_table[data[i]];
        }
    }
}

extern "C"
nmc_error_t nmc_transform_u8(nmc_region_t *region,
                             size_t offset,
                             size_t size,
                             const uint8_t table[256],
                             nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    uint8_t *d_table;
    uint8_t *data;
    
    if (!region || !table)
        return NMC_ERROR_INVALID_ARG;
    
    if (offset + size > nmc_size(region))
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    data = (uint8_t *)nmc_gpu_ptr(region) + offset;
    
    /* Upload lookup table */
    cudaMalloc(&d_table, 256);
    cudaMemcpyAsync(d_table, table, 256, cudaMemcpyHostToDevice, cuda_stream);
    
    /* Launch kernel */
    size_t threads_needed = (size + 3) / 4;
    dim3 block(BLOCK_SIZE);
    dim3 grid((threads_needed + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    kernel_transform_u8<<<grid, block, 0, cuda_stream>>>(data, size, d_table);
    
    cudaFree(d_table);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    return NMC_SUCCESS;
}

/*
 * ============================================================
 * Search Kernels
 * ============================================================
 */

/*
 * Pattern search kernel
 *
 * Uses parallel string matching with early termination.
 * Each thread checks one starting position.
 */
__global__ void kernel_search(const uint8_t *data,
                              size_t data_size,
                              const uint8_t *pattern,
                              size_t pattern_len,
                              uint64_t *results,
                              size_t max_results,
                              uint32_t *result_count)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx + pattern_len > data_size)
        return;
    
    /* Check for match at this position */
    bool match = true;
    for (size_t i = 0; i < pattern_len && match; i++) {
        if (data[idx + i] != pattern[i]) {
            match = false;
        }
    }
    
    if (match) {
        /* Atomic increment and store result */
        uint32_t slot = atomicAdd(result_count, 1);
        if (slot < max_results) {
            results[slot] = idx;
        }
    }
}

extern "C"
nmc_error_t nmc_search(nmc_region_t *region,
                       size_t offset,
                       size_t size,
                       const void *pattern,
                       size_t pattern_len,
                       nmc_region_t *results,
                       size_t max_results,
                       uint64_t *count_out,
                       nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    uint8_t *d_pattern;
    uint32_t *d_count;
    uint32_t h_count;
    
    if (!region || !pattern || !results || !count_out)
        return NMC_ERROR_INVALID_ARG;
    
    if (offset + size > nmc_size(region))
        return NMC_ERROR_INVALID_ARG;
    
    if (pattern_len == 0 || pattern_len > size)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    
    /* Upload pattern and allocate counter */
    cudaMalloc(&d_pattern, pattern_len);
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemcpyAsync(d_pattern, pattern, pattern_len, 
                    cudaMemcpyHostToDevice, cuda_stream);
    cudaMemsetAsync(d_count, 0, sizeof(uint32_t), cuda_stream);
    
    /* Launch search */
    size_t search_range = size - pattern_len + 1;
    dim3 block(BLOCK_SIZE);
    dim3 grid((search_range + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    kernel_search<<<grid, block, 0, cuda_stream>>>(
        (uint8_t *)nmc_gpu_ptr(region) + offset,
        size,
        d_pattern,
        pattern_len,
        (uint64_t *)nmc_gpu_ptr(results),
        max_results,
        d_count
    );
    
    /* Get result count */
    cudaMemcpyAsync(&h_count, d_count, sizeof(uint32_t),
                    cudaMemcpyDeviceToHost, cuda_stream);
    cudaStreamSynchronize(cuda_stream);
    
    *count_out = h_count;
    
    cudaFree(d_pattern);
    cudaFree(d_count);
    
    return NMC_SUCCESS;
}

/*
 * ============================================================
 * Reduction Kernels
 * ============================================================
 */

/*
 * Sum reduction using CUB
 */
extern "C"
nmc_error_t nmc_reduce_sum_f32(nmc_region_t *region,
                               size_t offset,
                               size_t count,
                               float *sum_out,
                               nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    float *d_out;
    void *d_temp = NULL;
    size_t temp_bytes = 0;
    
    if (!region || !sum_out)
        return NMC_ERROR_INVALID_ARG;
    
    if (offset + count * sizeof(float) > nmc_size(region))
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    float *data = (float *)((char *)nmc_gpu_ptr(region) + offset);
    
    /* Allocate output */
    cudaMalloc(&d_out, sizeof(float));
    
    /* Get temp storage size */
    cub::DeviceReduce::Sum(d_temp, temp_bytes, data, d_out, count, cuda_stream);
    
    /* Allocate temp storage */
    cudaMalloc(&d_temp, temp_bytes);
    
    /* Run reduction */
    cub::DeviceReduce::Sum(d_temp, temp_bytes, data, d_out, count, cuda_stream);
    
    /* Copy result back */
    cudaMemcpyAsync(sum_out, d_out, sizeof(float), 
                    cudaMemcpyDeviceToHost, cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    cudaFree(d_temp);
    cudaFree(d_out);
    
    return NMC_SUCCESS;
}

/*
 * Max reduction with index
 */
struct ArgMax {
    float value;
    uint64_t index;
};

__device__ __forceinline__ ArgMax argmax_op(ArgMax a, ArgMax b) {
    return (a.value > b.value) ? a : b;
}

__global__ void kernel_argmax(const float *data,
                              size_t count,
                              ArgMax *result)
{
    __shared__ ArgMax shared[BLOCK_SIZE];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    /* Initialize with identity */
    ArgMax local;
    local.value = -INFINITY;
    local.index = 0;
    
    /* Each thread finds local max */
    for (size_t i = idx; i < count; i += blockDim.x * gridDim.x) {
        if (data[i] > local.value) {
            local.value = data[i];
            local.index = i;
        }
    }
    
    shared[threadIdx.x] = local;
    __syncthreads();
    
    /* Block reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared[threadIdx.x] = argmax_op(shared[threadIdx.x], 
                                            shared[threadIdx.x + s]);
        }
        __syncthreads();
    }
    
    /* Write block result */
    if (threadIdx.x == 0) {
        atomicMax((int *)&result->value, __float_as_int(shared[0].value));
        if (shared[0].value == result->value) {
            result->index = shared[0].index;
        }
    }
}

extern "C"
nmc_error_t nmc_reduce_max_f32(nmc_region_t *region,
                               size_t offset,
                               size_t count,
                               float *max_out,
                               uint64_t *idx_out,
                               nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    ArgMax *d_result;
    ArgMax h_result;
    
    if (!region || !max_out)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    float *data = (float *)((char *)nmc_gpu_ptr(region) + offset);
    
    cudaMalloc(&d_result, sizeof(ArgMax));
    h_result.value = -INFINITY;
    h_result.index = 0;
    cudaMemcpyAsync(d_result, &h_result, sizeof(ArgMax),
                    cudaMemcpyHostToDevice, cuda_stream);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(min((size_t)1024, (count + BLOCK_SIZE - 1) / BLOCK_SIZE));
    
    kernel_argmax<<<grid, block, 0, cuda_stream>>>(data, count, d_result);
    
    cudaMemcpyAsync(&h_result, d_result, sizeof(ArgMax),
                    cudaMemcpyDeviceToHost, cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    *max_out = h_result.value;
    if (idx_out) {
        *idx_out = h_result.index;
    }
    
    cudaFree(d_result);
    
    return NMC_SUCCESS;
}

/*
 * ============================================================
 * Histogram Kernel
 * ============================================================
 */

__global__ void kernel_histogram(const uint8_t *data,
                                 size_t size,
                                 uint64_t *histogram)
{
    __shared__ uint32_t shared_hist[256];
    
    /* Initialize shared histogram */
    if (threadIdx.x < 256) {
        shared_hist[threadIdx.x] = 0;
    }
    __syncthreads();
    
    /* Count in shared memory */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    
    for (size_t i = idx; i < size; i += stride) {
        atomicAdd(&shared_hist[data[i]], 1);
    }
    __syncthreads();
    
    /* Merge to global histogram */
    if (threadIdx.x < 256) {
        atomicAdd((unsigned long long *)&histogram[threadIdx.x], 
                  shared_hist[threadIdx.x]);
    }
}

extern "C"
nmc_error_t nmc_histogram_u8(nmc_region_t *region,
                             size_t offset,
                             size_t size,
                             uint64_t hist_out[256],
                             nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    uint64_t *d_hist;
    
    if (!region || !hist_out)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    uint8_t *data = (uint8_t *)nmc_gpu_ptr(region) + offset;
    
    cudaMalloc(&d_hist, 256 * sizeof(uint64_t));
    cudaMemsetAsync(d_hist, 0, 256 * sizeof(uint64_t), cuda_stream);
    
    dim3 block(BLOCK_SIZE);
    dim3 grid(min((size_t)1024, (size + BLOCK_SIZE - 1) / BLOCK_SIZE));
    
    kernel_histogram<<<grid, block, 0, cuda_stream>>>(data, size, d_hist);
    
    /* Only 256 values cross PCIe! */
    cudaMemcpyAsync(hist_out, d_hist, 256 * sizeof(uint64_t),
                    cudaMemcpyDeviceToHost, cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    cudaFree(d_hist);
    
    return NMC_SUCCESS;
}

/*
 * ============================================================
 * Sort Kernels (using CUB)
 * ============================================================
 */

extern "C"
nmc_error_t nmc_sort_u32(nmc_region_t *region,
                         size_t offset,
                         size_t count,
                         nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    void *d_temp = NULL;
    size_t temp_bytes = 0;
    uint32_t *d_out;
    
    if (!region)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    uint32_t *data = (uint32_t *)((char *)nmc_gpu_ptr(region) + offset);
    
    /* CUB requires double buffer - allocate output */
    cudaMalloc(&d_out, count * sizeof(uint32_t));
    
    /* Get temp storage size */
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, data, d_out, count, 
                                   0, 32, cuda_stream);
    cudaMalloc(&d_temp, temp_bytes);
    
    /* Sort */
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, data, d_out, count,
                                   0, 32, cuda_stream);
    
    /* Copy back (within VRAM!) */
    cudaMemcpyAsync(data, d_out, count * sizeof(uint32_t),
                    cudaMemcpyDeviceToDevice, cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    cudaFree(d_temp);
    cudaFree(d_out);
    
    return NMC_SUCCESS;
}

extern "C"
nmc_error_t nmc_sort_f32(nmc_region_t *region,
                         size_t offset,
                         size_t count,
                         nmc_stream_t *stream)
{
    cudaStream_t cuda_stream;
    void *d_temp = NULL;
    size_t temp_bytes = 0;
    float *d_out;
    
    if (!region)
        return NMC_ERROR_INVALID_ARG;
    
    cuda_stream = stream ? stream->cuda_stream : 0;
    float *data = (float *)((char *)nmc_gpu_ptr(region) + offset);
    
    cudaMalloc(&d_out, count * sizeof(float));
    
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, data, d_out, count,
                                   0, 32, cuda_stream);
    cudaMalloc(&d_temp, temp_bytes);
    
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, data, d_out, count,
                                   0, 32, cuda_stream);
    
    cudaMemcpyAsync(data, d_out, count * sizeof(float),
                    cudaMemcpyDeviceToDevice, cuda_stream);
    
    if (!stream) {
        cudaStreamSynchronize(cuda_stream);
    }
    
    cudaFree(d_temp);
    cudaFree(d_out);
    
    return NMC_SUCCESS;
}
