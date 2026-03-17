/*
 * omega_kernels.cu — CUDA Kernels for ΩMEGA Coupling Computation
 *
 * Sitton Number arithmetic on GPU: each thread handles one qubit pair.
 * Path signatures pre-computed, stored in VRAM. Coalesced access via
 * packed sitton_f32 struct (8 bytes, aligned).
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

/* ── Device-side Sitton arithmetic ───────────────────────────── */

struct sitton_f32 {
    float real;
    float prob;
};

__device__ __forceinline__
sitton_f32 sn_mul_d(sitton_f32 a, sitton_f32 b)
{
    return { a.real * b.real,
             a.real * b.prob + a.prob * b.real };
}

__device__ __forceinline__
sitton_f32 sn_add_d(sitton_f32 a, sitton_f32 b)
{
    return { a.real + b.real, a.prob + b.prob };
}

__device__ __forceinline__
sitton_f32 sn_exp_d(sitton_f32 x)
{
    float ea = expf(x.real);
    return { ea, x.prob * ea };
}

__device__ __forceinline__
sitton_f32 sn_scale_d(sitton_f32 x, float alpha)
{
    return { x.real * alpha, x.prob * alpha };
}

/* ── Coupling Kernel ─────────────────────────────────────────── */

__global__ void omega_coupling_kernel(
    const uint64_t  *path_signatures,
    const sitton_f32 *material_eps,
    const float     *segment_lengths,
    const uint32_t  *pair_seg_offsets,
    const uint32_t  *pair_seg_indices,
    sitton_f32      *coupling_out,
    sitton_f32      *phase_out,
    uint32_t         num_pairs)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    uint32_t seg_start = pair_seg_offsets[idx];
    uint32_t seg_end   = pair_seg_offsets[idx + 1];

    /* Accumulate coupling along path using Sitton propagation.
     * exp(-(alpha_r + pfrak*alpha_p)*len) =
     *   exp(-alpha_r*len) * (1 + pfrak*(-alpha_p*len))
     */
    sitton_f32 total_atten = {1.0f, 0.0f};
    sitton_f32 total_phase = {0.0f, 0.0f};

    for (uint32_t s = seg_start; s < seg_end; s++) {
        uint32_t seg_idx = pair_seg_indices[s];
        sitton_f32 eps = material_eps[seg_idx];
        float len = segment_lengths[seg_idx];

        /* Propagation: exp(-(eps)*len) in Sitton arithmetic */
        sitton_f32 neg_eps_len = { -eps.real * len, -eps.prob * len };
        sitton_f32 seg_atten = sn_exp_d(neg_eps_len);

        total_atten = sn_mul_d(total_atten, seg_atten);

        /* Phase accumulation: 2*pi*f*sqrt(eps)*len / c */
        float phase_inc = 2.0f * 3.14159265f * sqrtf(fabsf(eps.real)) * len * 1e-6f / 3e8f;
        sitton_f32 dp = {phase_inc, eps.prob * 0.5f / (sqrtf(fabsf(eps.real)) + 1e-30f) * len};
        total_phase = sn_add_d(total_phase, dp);
    }

    coupling_out[idx] = total_atten;
    phase_out[idx]    = total_phase;
}

/* ── Ensemble Accumulation Kernel ────────────────────────────── */

__global__ void omega_ensemble_accum_kernel(
    const sitton_f32 *coupling_batch,
    sitton_f32       *accum_sum,
    sitton_f32       *accum_sum_sq,
    float            *worst_case,
    float            *best_case,
    uint32_t         *fail_count,
    uint32_t          num_pairs,
    float             fail_threshold)
{
    uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_pairs) return;

    sitton_f32 val = coupling_batch[idx];

    /* Atomic-free accumulation: each realization writes to its own slot,
     * reduction done on CPU. For simplicity here, use atomicAdd. */
    atomicAdd(&accum_sum[idx].real, val.real);
    atomicAdd(&accum_sum[idx].prob, val.prob);

    sitton_f32 sq = sn_mul_d(val, val);
    atomicAdd(&accum_sum_sq[idx].real, sq.real);
    atomicAdd(&accum_sum_sq[idx].prob, sq.prob);

    /* Track extremes */
    atomicMax((int *)&worst_case[idx], __float_as_int(fabsf(val.real)));
    atomicMin((int *)&best_case[idx],  __float_as_int(fabsf(val.real)));

    if (fabsf(val.real) > fail_threshold)
        atomicAdd(&fail_count[idx], 1);
}

/* ── Host wrappers ───────────────────────────────────────────── */

extern "C" {

void omega_launch_coupling_kernel(
    const uint64_t *sigs, const void *eps, const float *lens,
    const uint32_t *offsets, const uint32_t *indices,
    void *coupling_out, void *phase_out,
    uint32_t npairs)
{
    dim3 block(256);
    dim3 grid((npairs + 255) / 256);
    omega_coupling_kernel<<<grid, block>>>(
        sigs, (const sitton_f32 *)eps, lens, offsets, indices,
        (sitton_f32 *)coupling_out, (sitton_f32 *)phase_out, npairs);
    cudaDeviceSynchronize();
}

void omega_launch_ensemble_accum(
    const void *batch, void *sum, void *sum_sq,
    float *worst, float *best, uint32_t *fail,
    uint32_t npairs, float threshold)
{
    dim3 block(256);
    dim3 grid((npairs + 255) / 256);
    omega_ensemble_accum_kernel<<<grid, block>>>(
        (const sitton_f32 *)batch, (sitton_f32 *)sum, (sitton_f32 *)sum_sq,
        worst, best, fail, npairs, threshold);
    cudaDeviceSynchronize();
}

} /* extern "C" */

