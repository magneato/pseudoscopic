#ifndef OMEGA_VRAM_H
#define OMEGA_VRAM_H
/*
 * omega_vram.h — VRAM Layout via Pseudoscopic Near-Memory
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <stdint.h>
#include <stddef.h>
#include "omega_geometry.h"
#include "omega_qxor.h"
#include "nearmem.h"
#include "omega_sitton.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    sitton_f32_t    coupling_mean;
    sitton_f32_t    coupling_variance;
    float           worst_case_xtalk;
    float           best_case_xtalk;
    uint32_t        failure_count;
} omega_ensemble_result_t;

typedef struct {
    nearmem_ctx_t       nm_ctx;
    nearmem_region_t    geometry_region;     /*  16 MB */
    nearmem_region_t    spline_region;       /*   4 MB */
    nearmem_region_t    coupling_region;     /*  32 MB */
    nearmem_region_t    ensemble_region;     /*  64 MB */
    nearmem_region_t    entropy_region;      /*  16 MB */
    nearmem_region_t    scratch_region;      /* 128 MB */
} omega_vram_layout_t;

int omega_vram_init(omega_vram_layout_t *vram, int gpu_index);
void omega_compute_couplings_tiled(omega_vram_layout_t *vram,
                                    const omega_chip_geometry_t *geom,
                                    omega_coupling_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_VRAM_H */

