#ifndef OMEGA_SPLINE_H
#define OMEGA_SPLINE_H
/*
 * omega_spline.h — Neural Spline™ Field Compression
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  NOTICE: Neural Splines™ is proprietary technology of        ║
 * ║  Neural Splines LLC. US Provisional Patent Application Filed.║
 * ║  (c) 2026 Robert L. Sitton, Jr. — All Rights Reserved       ║
 * ║                                                               ║
 * ║  Unauthorized reproduction or derivative works prohibited.    ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * Key insight: electromagnetic fields in quantum chips live on a
 * low-dimensional manifold constrained by boundary conditions,
 * waveguide modes, resonant structure, and exponential decay.
 * Neural Splines find this manifold via hierarchical B-splines.
 *
 * Hierarchy:
 *   Level 0 (coarsest): global structure
 *   Level 1+: residual refinement, each 2× finer
 *   Each level stores the RESIDUAL after coarser levels.
 *
 * Tiling: symmetric chip regions share control points.
 *   800 instances × 20 unique types = 40:1 before compression.
 */

#include <stdint.h>
#include <stddef.h>
#include "omega_geometry.h"
#include "omega_sitton.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Cubic B-spline basis (Cox-de Boor, uniform knots) ──────── */

static inline void ns_basis_cubic(float t, float b[4]) {
    float t2 = t * t, t3 = t2 * t;
    float u = 1.0f - t, u2 = u * u, u3 = u2 * u;
    b[0] = u3 / 6.0f;
    b[1] = (3.0f * t3 - 6.0f * t2 + 4.0f) / 6.0f;
    b[2] = (-3.0f * t3 + 3.0f * t2 + 3.0f * t + 1.0f) / 6.0f;
    b[3] = t3 / 6.0f;
}

static inline void ns_basis_cubic_deriv(float t, float db[4]) {
    float t2 = t * t;
    db[0] = -0.5f * (1.0f - t) * (1.0f - t);
    db[1] = 1.5f * t2 - 2.0f * t;
    db[2] = -1.5f * t2 + t + 0.5f;
    db[3] = 0.5f * t2;
}

/* ── 3D Neural Spline ────────────────────────────────────────── */

typedef struct {
    float      *ctrl;               /* [nx × ny × nz] flattened     */
    size_t      nx, ny, nz;
    float       x_min, x_max, y_min, y_max, z_min, z_max;
    float       inv_dx, inv_dy, inv_dz;
} ns_spline_3d_t;

int   ns_spline_3d_init(ns_spline_3d_t *s, size_t nx, size_t ny, size_t nz,
                        float xn, float xx, float yn, float yx, float zn, float zx);
void  ns_spline_3d_destroy(ns_spline_3d_t *s);
float ns_spline_3d_eval(const ns_spline_3d_t *s, float x, float y, float z);
float ns_spline_3d_eval_grad(const ns_spline_3d_t *s, float x, float y, float z, float g[3]);

sitton_f32_t ns_spline_3d_eval_sitton(const ns_spline_3d_t *s,
    sitton_f32_t x, sitton_f32_t y, sitton_f32_t z);

/* ── Hierarchical field: multi-resolution compression ────────── */

#define NS_MAX_LEVELS 8

typedef struct {
    ns_spline_3d_t  levels[NS_MAX_LEVELS];
    uint8_t         num_levels;
    bool            valid[NS_MAX_LEVELS];
    size_t          original_bytes;
    size_t          compressed_bytes;
    float           rms_residual;
    float           max_residual;
} ns_hierarchy_t;

int   ns_hierarchy_fit(ns_hierarchy_t *h, const float *data,
                       size_t w, size_t ht, size_t d,
                       float xn, float xx, float yn, float yx, float zn, float zx,
                       float target_rms, uint8_t max_levels, size_t base_ctrl);
float ns_hierarchy_eval(const ns_hierarchy_t *h, float x, float y, float z);
sitton_f32_t ns_hierarchy_eval_sitton(const ns_hierarchy_t *h,
    sitton_f32_t x, sitton_f32_t y, sitton_f32_t z);
void  ns_hierarchy_destroy(ns_hierarchy_t *h);
void  ns_hierarchy_print_stats(const ns_hierarchy_t *h);

/* ── Tiled field: symmetry-aware chip representation ─────────── */

typedef struct {
    uint64_t        master_seed;
    float          *class_controls;
    size_t          num_classes;
    size_t          ctrl_per_class;
    float          *instance_deltas;
    size_t          delta_ctrl_per_instance;
    float          *coupling_controls;
    size_t          num_interfaces;
    size_t          coupling_ctrl_count;
} omega_spline_field_t;

void omega_field_eval(const omega_spline_field_t *field,
                      const omega_chip_geometry_t *geom,
                      float x, float y, float z,
                      float out_E[3], float out_H[3]);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_SPLINE_H */

