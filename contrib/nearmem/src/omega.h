#ifndef OMEGA_H
#define OMEGA_H
/*
 * omega.h — ΩMEGA: Orthoscopic Maxwell Electromagnetic GPU Accelerator
 *
 * Single-GPU quantum chip electromagnetic simulation.
 * Replaces 7,168-GPU brute-force FDTD with orthoscopic approach.
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_geometry.h"
#include "omega_sitton.h"
#include "omega_qxor.h"
#include "omega_spline.h"
#include "omega_entropy.h"
#include "omega_vram.h"
#include "omega_validate.h"
#include "omega_hashbelt.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OMEGA_VERSION "0.1.0"

typedef struct {
    const char     *gds_file;
    const char     *material_file;
    const char     *control_verilog;
    float           freq_min_ghz;
    float           freq_max_ghz;
    float           freq_step_ghz;
    float           fab_tolerance_nm;
    uint32_t        ensemble_size;
    int             gpu_index;
    const char     *entropy_device;
    const char     *seed_log_path;
    float           target_rms;
    uint8_t         max_spline_levels;
    size_t          base_ctrl_per_dim;
} omega_config_t;

static inline omega_config_t omega_config_default(void) {
    return (omega_config_t){
        .gds_file = NULL, .material_file = NULL, .control_verilog = NULL,
        .freq_min_ghz = 4.0f, .freq_max_ghz = 8.0f, .freq_step_ghz = 0.1f,
        .fab_tolerance_nm = 10.0f, .ensemble_size = 256,
        .gpu_index = 0, .entropy_device = NULL, .seed_log_path = "omega_seeds.bin",
        .target_rms = 1e-4f, .max_spline_levels = 4, .base_ctrl_per_dim = 8
    };
}

typedef struct {
    omega_coupling_graph_t      nominal;
    omega_ensemble_result_t    *ensemble;
    size_t                      num_pairs;
    float                      *s_parameters;
    sitton_f32_t               *sensitivity;
    uint32_t                   *fragile_couplings;
    size_t                      num_fragile;
    float                       overall_yield_estimate;
    double                      total_time_seconds;
    size_t                      vram_used_bytes;
    uint64_t                    entropy_consumed_bytes;
} omega_result_t;

int omega_simulate(const omega_config_t *config, omega_result_t *result);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_H */

