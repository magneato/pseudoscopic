/*
 * nearmem_kernels.cu - CUDA Kernels for Near-Memory Computing
 *
 * These kernels operate directly on VRAM data. No PCIe copies.
 * The data is already in GPU memory - we just compute on it.
 *
 * Key insight: When pseudoscopic maps VRAM as a block device,
 * and we mmap that device, we get CPU access to GPU memory.
 * CUDA can access the exact same memory via device pointers.
 *
 * Copyright (C) 2025 Neural Splines LLC
 * License: MIT
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

/* Block sizes tuned for modern GPUs */
#define BLOCK_SIZE 256
#define WARP_SIZE 32

/*
 * ============================================================
 * Search Kernels
 * ============================================================
 */

/*
 * find_byte_kernel - Find first occurrence of a byte
 *
 * Each thread checks a chunk of memory. First match wins.
 * Uses atomicMin to find globally first match.
 */
__global__ void find_byte_kernel(const uint8_t *data, 
                                  size_t size,
                                  uint8_t target,
                                  int64_t *result)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < size; i += stride) {
        if (data[i] == target) {
            atomicMin((unsigned long long*)result, (unsigned long long)i);
            return;  /* Early exit */
        }
    }
}

/*
 * find_pattern_kernel - Find pattern using parallel search
 *
 * Each thread checks one position. Pattern is in constant memory
 * for fast broadcast access.
 */
__constant__ uint8_t c_pattern[256];
__constant__ size_t c_pattern_len;

__global__ void find_pattern_kernel(const uint8_t *data,
                                     size_t size,
                                     int64_t *result)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t pattern_len = c_pattern_len;
    
    for (size_t i = idx; i <= size - pattern_len; i += stride) {
        bool match = true;
        for (size_t j = 0; j < pattern_len && match; j++) {
            if (data[i + j] != c_pattern[j])
                match = false;
        }
        if (match) {
            atomicMin((unsigned long long*)result, (unsigned long long)i);
            return;
        }
    }
}

/*
 * count_pattern_kernel - Count pattern occurrences
 *
 * Uses shared memory for partial counts, then atomic add.
 */
__global__ void count_pattern_kernel(const uint8_t *data,
                                      size_t size,
                                      uint64_t *count)
{
    __shared__ uint32_t shared_count;
    
    if (threadIdx.x == 0)
        shared_count = 0;
    __syncthreads();
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    size_t pattern_len = c_pattern_len;
    uint32_t local_count = 0;
    
    for (size_t i = idx; i <= size - pattern_len; i += stride) {
        bool match = true;
        for (size_t j = 0; j < pattern_len && match; j++) {
            if (data[i + j] != c_pattern[j])
                match = false;
        }
        if (match)
            local_count++;
    }
    
    atomicAdd(&shared_count, local_count);
    __syncthreads();
    
    if (threadIdx.x == 0)
        atomicAdd((unsigned long long*)count, shared_count);
}

/*
 * ============================================================
 * Transform Kernels
 * ============================================================
 */

/*
 * transform_lut_kernel - Apply lookup table transformation
 *
 * Each byte b becomes lut[b]. LUT is in constant memory.
 */
__constant__ uint8_t c_lut[256];

__global__ void transform_lut_kernel(uint8_t *data, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    /* Process 4 bytes at a time for better memory throughput */
    for (size_t i = idx * 4; i < size; i += stride * 4) {
        if (i + 3 < size) {
            uint32_t *ptr = (uint32_t*)(data + i);
            uint32_t val = *ptr;
            uint32_t out;
            
            out  = c_lut[(val >>  0) & 0xFF] <<  0;
            out |= c_lut[(val >>  8) & 0xFF] <<  8;
            out |= c_lut[(val >> 16) & 0xFF] << 16;
            out |= c_lut[(val >> 24) & 0xFF] << 24;
            
            *ptr = out;
        } else {
            /* Handle tail bytes */
            for (size_t j = i; j < size; j++)
                data[j] = c_lut[data[j]];
        }
    }
}

/*
 * to_uppercase_kernel - Optimized case conversion
 *
 * Faster than generic LUT for this specific transform.
 */
__global__ void to_uppercase_kernel(uint8_t *data, size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < size; i += stride) {
        uint8_t c = data[i];
        if (c >= 'a' && c <= 'z')
            data[i] = c - 32;
    }
}

/*
 * ============================================================
 * Reduction Kernels
 * ============================================================
 */

/*
 * histogram_kernel - Compute byte histogram
 *
 * Uses shared memory histograms per block, then atomic merge.
 */
__global__ void histogram_kernel(const uint8_t *data,
                                  size_t size,
                                  uint64_t *histogram)
{
    __shared__ uint32_t shared_hist[256];
    
    /* Initialize shared histogram */
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        shared_hist[i] = 0;
    __syncthreads();
    
    /* Count into shared memory */
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < size; i += stride)
        atomicAdd(&shared_hist[data[i]], 1);
    
    __syncthreads();
    
    /* Merge to global histogram */
    for (int i = threadIdx.x; i < 256; i += blockDim.x)
        atomicAdd((unsigned long long*)&histogram[i], shared_hist[i]);
}

/*
 * reduce_sum_f32_kernel - Parallel sum reduction
 *
 * Two-phase: block-level reduction, then grid-level.
 */
__global__ void reduce_sum_f32_kernel(const float *data,
                                       size_t count,
                                       float *partial_sums)
{
    __shared__ float shared_data[BLOCK_SIZE];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    /* Load and accumulate across grid */
    float sum = 0;
    for (size_t i = idx; i < count; i += stride)
        sum += data[i];
    
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    
    /* Block-level reduction */
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        __syncthreads();
    }
    
    /* Write block result */
    if (threadIdx.x == 0)
        partial_sums[blockIdx.x] = shared_data[0];
}

/*
 * reduce_minmax_u32_kernel - Find min and max in parallel
 */
__global__ void reduce_minmax_u32_kernel(const uint32_t *data,
                                          size_t count,
                                          uint32_t *min_out,
                                          uint32_t *max_out)
{
    __shared__ uint32_t shared_min[BLOCK_SIZE];
    __shared__ uint32_t shared_max[BLOCK_SIZE];
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    uint32_t local_min = UINT32_MAX;
    uint32_t local_max = 0;
    
    for (size_t i = idx; i < count; i += stride) {
        uint32_t val = data[i];
        if (val < local_min) local_min = val;
        if (val > local_max) local_max = val;
    }
    
    shared_min[threadIdx.x] = local_min;
    shared_max[threadIdx.x] = local_max;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            if (shared_min[threadIdx.x + s] < shared_min[threadIdx.x])
                shared_min[threadIdx.x] = shared_min[threadIdx.x + s];
            if (shared_max[threadIdx.x + s] > shared_max[threadIdx.x])
                shared_max[threadIdx.x] = shared_max[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    if (threadIdx.x == 0) {
        atomicMin(min_out, shared_min[0]);
        atomicMax(max_out, shared_max[0]);
    }
}

/*
 * ============================================================
 * Sort Kernels (Radix Sort)
 * ============================================================
 */

/*
 * radix_sort_histogram_kernel - Count digit occurrences
 */
__global__ void radix_sort_histogram_kernel(const uint32_t *data,
                                             size_t count,
                                             uint32_t *histograms,
                                             int bit_offset)
{
    __shared__ uint32_t shared_hist[16];  /* 4 bits = 16 buckets */
    
    if (threadIdx.x < 16)
        shared_hist[threadIdx.x] = 0;
    __syncthreads();
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        uint32_t digit = (data[i] >> bit_offset) & 0xF;
        atomicAdd(&shared_hist[digit], 1);
    }
    __syncthreads();
    
    if (threadIdx.x < 16)
        atomicAdd(&histograms[blockIdx.x * 16 + threadIdx.x], shared_hist[threadIdx.x]);
}

/*
 * radix_sort_scatter_kernel - Scatter elements to sorted positions
 */
__global__ void radix_sort_scatter_kernel(const uint32_t *input,
                                           uint32_t *output,
                                           const uint32_t *offsets,
                                           size_t count,
                                           int bit_offset)
{
    __shared__ uint32_t shared_offsets[16];
    
    if (threadIdx.x < 16)
        shared_offsets[threadIdx.x] = offsets[blockIdx.x * 16 + threadIdx.x];
    __syncthreads();
    
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < count) {
        uint32_t val = input[idx];
        uint32_t digit = (val >> bit_offset) & 0xF;
        uint32_t pos = atomicAdd(&shared_offsets[digit], 1);
        output[pos] = val;
    }
}

/*
 * ============================================================
 * Log Processing Kernel (The Killer App Demo)
 * ============================================================
 */

/*
 * grep_kernel - Find all lines matching pattern
 *
 * This is the showcase: parallel grep on GPU without copying logs.
 * Each warp handles a chunk of the file.
 *
 * Output: array of line numbers containing matches
 */
__global__ void grep_kernel(const char *data,
                            size_t size,
                            const char *pattern,
                            size_t pattern_len,
                            uint32_t *match_lines,
                            uint32_t *match_count,
                            uint32_t max_matches)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    uint32_t line_num = 0;
    size_t line_start = 0;
    
    /* Find our starting line number */
    for (size_t i = 0; i < idx && i < size; i++) {
        if (data[i] == '\n') {
            line_num++;
            line_start = i + 1;
        }
    }
    
    /* Search our portion */
    for (size_t i = idx; i <= size - pattern_len; i += stride) {
        /* Update line tracking */
        while (line_start < i) {
            if (data[line_start] == '\n') {
                line_num++;
            }
            line_start++;
        }
        
        /* Check for pattern match */
        bool match = true;
        for (size_t j = 0; j < pattern_len && match; j++) {
            if (data[i + j] != pattern[j])
                match = false;
        }
        
        if (match) {
            uint32_t idx = atomicAdd(match_count, 1);
            if (idx < max_matches)
                match_lines[idx] = line_num;
        }
    }
}

/*
 * ============================================================
 * Neural Network Helper Kernels
 * ============================================================
 */

/*
 * quantize_f32_to_u8_kernel - Quantize float weights to uint8
 *
 * For Neural Splines inference: compress weights in-place.
 */
__global__ void quantize_f32_to_u8_kernel(const float *input,
                                           uint8_t *output,
                                           size_t count,
                                           float scale,
                                           float zero_point)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        float val = input[i] * scale + zero_point;
        val = fminf(255.0f, fmaxf(0.0f, val));
        output[i] = (uint8_t)val;
    }
}

/*
 * dequantize_u8_to_f32_kernel - Expand quantized weights
 */
__global__ void dequantize_u8_to_f32_kernel(const uint8_t *input,
                                             float *output,
                                             size_t count,
                                             float scale,
                                             float zero_point)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < count; i += stride) {
        output[i] = ((float)input[i] - zero_point) / scale;
    }
}

/*
 * ============================================================
 * Host-side Launcher Functions
 * ============================================================
 */

extern "C" {

int nearmem_cuda_find_byte(void *data, size_t size, uint8_t target, int64_t *result)
{
    int64_t *d_result;
    int64_t h_result = INT64_MAX;
    
    cudaMalloc(&d_result, sizeof(int64_t));
    cudaMemcpy(d_result, &h_result, sizeof(int64_t), cudaMemcpyHostToDevice);
    
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks, 65535);
    
    find_byte_kernel<<<blocks, BLOCK_SIZE>>>((uint8_t*)data, size, target, d_result);
    
    cudaMemcpy(&h_result, d_result, sizeof(int64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
    
    *result = (h_result == INT64_MAX) ? -1 : h_result;
    return 0;
}

int nearmem_cuda_histogram(void *data, size_t size, uint64_t *histogram)
{
    uint64_t *d_histogram;
    
    cudaMalloc(&d_histogram, 256 * sizeof(uint64_t));
    cudaMemset(d_histogram, 0, 256 * sizeof(uint64_t));
    
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks, 65535);
    
    histogram_kernel<<<blocks, BLOCK_SIZE>>>((uint8_t*)data, size, d_histogram);
    
    cudaMemcpy(histogram, d_histogram, 256 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    cudaFree(d_histogram);
    
    return 0;
}

int nearmem_cuda_transform_lut(void *data, size_t size, const uint8_t *lut)
{
    cudaMemcpyToSymbol(c_lut, lut, 256);
    
    int blocks = (size + BLOCK_SIZE * 4 - 1) / (BLOCK_SIZE * 4);
    blocks = min(blocks, 65535);
    
    transform_lut_kernel<<<blocks, BLOCK_SIZE>>>((uint8_t*)data, size);
    cudaDeviceSynchronize();
    
    return 0;
}

int nearmem_cuda_reduce_sum_f32(void *data, size_t count, float *result)
{
    int blocks = min((int)((count + BLOCK_SIZE - 1) / BLOCK_SIZE), 1024);
    
    float *d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(float));
    
    reduce_sum_f32_kernel<<<blocks, BLOCK_SIZE>>>((float*)data, count, d_partial);
    
    /* Final reduction on CPU (for simplicity) */
    float *h_partial = (float*)malloc(blocks * sizeof(float));
    cudaMemcpy(h_partial, d_partial, blocks * sizeof(float), cudaMemcpyDeviceToHost);
    
    float sum = 0;
    for (int i = 0; i < blocks; i++)
        sum += h_partial[i];
    
    *result = sum;
    
    free(h_partial);
    cudaFree(d_partial);
    
    return 0;
}

int nearmem_cuda_grep(void *data, size_t size, 
                      const char *pattern, size_t pattern_len,
                      uint32_t *match_lines, uint32_t *match_count,
                      uint32_t max_matches)
{
    char *d_pattern;
    uint32_t *d_match_lines, *d_match_count;
    
    cudaMalloc(&d_pattern, pattern_len);
    cudaMalloc(&d_match_lines, max_matches * sizeof(uint32_t));
    cudaMalloc(&d_match_count, sizeof(uint32_t));
    
    cudaMemcpy(d_pattern, pattern, pattern_len, cudaMemcpyHostToDevice);
    cudaMemset(d_match_count, 0, sizeof(uint32_t));
    
    int blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    blocks = min(blocks, 65535);
    
    grep_kernel<<<blocks, BLOCK_SIZE>>>((char*)data, size, d_pattern, pattern_len,
                                         d_match_lines, d_match_count, max_matches);
    
    cudaMemcpy(match_count, d_match_count, sizeof(uint32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(match_lines, d_match_lines, 
               min(*match_count, max_matches) * sizeof(uint32_t), 
               cudaMemcpyDeviceToHost);
    
    cudaFree(d_pattern);
    cudaFree(d_match_lines);
    cudaFree(d_match_count);
    
    return 0;
}

} /* extern "C" */
