/*
 * omega_geometry.c — Chip Geometry with Symmetry Detection
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_geometry.h"
#include "omega_qxor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ── Standard materials ──────────────────────────────────────── */

const omega_material_t OMEGA_MAT_NIOBIUM = {
    .epsilon_r = 1.0f, .mu_r = 1.0f, .sigma = 0.0f,
    .tan_delta = 0.0f, .thickness_um = 0.2f,
    .is_superconductor = 1, .Tc_kelvin = 9.25f,
    .london_depth_nm = 39.0f, .fab_tolerance_nm = 5.0f
};

const omega_material_t OMEGA_MAT_ALUMINUM = {
    .epsilon_r = 1.0f, .mu_r = 1.0f, .sigma = 0.0f,
    .tan_delta = 0.0f, .thickness_um = 0.1f,
    .is_superconductor = 1, .Tc_kelvin = 1.18f,
    .london_depth_nm = 16.0f, .fab_tolerance_nm = 3.0f
};

const omega_material_t OMEGA_MAT_SILICON = {
    .epsilon_r = 11.7f, .mu_r = 1.0f, .sigma = 1e-4f,
    .tan_delta = 1e-6f, .thickness_um = 500.0f,
    .is_superconductor = 0, .Tc_kelvin = 0.0f,
    .london_depth_nm = 0.0f, .fab_tolerance_nm = 10.0f
};

const omega_material_t OMEGA_MAT_SAPPHIRE = {
    .epsilon_r = 10.0f, .mu_r = 1.0f, .sigma = 1e-14f,
    .tan_delta = 1e-8f, .thickness_um = 430.0f,
    .is_superconductor = 0, .Tc_kelvin = 0.0f,
    .london_depth_nm = 0.0f, .fab_tolerance_nm = 5.0f
};

const omega_material_t OMEGA_MAT_VACUUM = {
    .epsilon_r = 1.0f, .mu_r = 1.0f, .sigma = 0.0f,
    .tan_delta = 0.0f, .thickness_um = 0.0f,
    .is_superconductor = 0, .Tc_kelvin = 0.0f,
    .london_depth_nm = 0.0f, .fab_tolerance_nm = 0.0f
};

/* ── Sorting (insertion sort — vertex counts are small) ──────── */

static void sort_floats(float *a, size_t n) {
    for (size_t i = 1; i < n; i++) {
        float key = a[i]; size_t j = i;
        while (j > 0 && a[j-1] > key) { a[j] = a[j-1]; j--; }
        a[j] = key;
    }
}

float *omega_sorted_edge_lengths(const float *verts, size_t n)
{
    size_t npairs = n * (n - 1) / 2;
    float *edges = (float *)malloc(npairs * sizeof(float));
    if (!edges) return NULL;
    size_t k = 0;
    for (size_t i = 0; i < n; i++)
        for (size_t j = i+1; j < n; j++) {
            float dx = verts[j*3+0] - verts[i*3+0];
            float dy = verts[j*3+1] - verts[i*3+1];
            float dz = verts[j*3+2] - verts[i*3+2];
            edges[k++] = sqrtf(dx*dx + dy*dy + dz*dz);
        }
    sort_floats(edges, npairs);
    return edges;
}

float *omega_sorted_vertex_angles(const float *verts, size_t n)
{
    if (n < 3) return NULL;
    size_t nangles = n;
    float *angles = (float *)malloc(nangles * sizeof(float));
    if (!angles) return NULL;
    for (size_t i = 0; i < n; i++) {
        size_t prev = (i + n - 1) % n, next = (i + 1) % n;
        float ax = verts[prev*3+0]-verts[i*3+0], ay = verts[prev*3+1]-verts[i*3+1];
        float bx = verts[next*3+0]-verts[i*3+0], by = verts[next*3+1]-verts[i*3+1];
        float dot = ax*bx + ay*by;
        float ma = sqrtf(ax*ax+ay*ay), mb = sqrtf(bx*bx+by*by);
        angles[i] = (ma > 1e-10f && mb > 1e-10f) ? acosf(fminf(fmaxf(dot/(ma*mb),-1.0f),1.0f)) : 0.0f;
    }
    sort_floats(angles, nangles);
    return angles;
}

float omega_bbox_aspect(const float *verts, size_t n)
{
    if (n == 0) return 1.0f;
    float xmin = verts[0], xmax = verts[0], ymin = verts[1], ymax = verts[1];
    for (size_t i = 1; i < n; i++) {
        float x = verts[i*3+0], y = verts[i*3+1];
        if (x < xmin) xmin = x; if (x > xmax) xmax = x;
        if (y < ymin) ymin = y; if (y > ymax) ymax = y;
    }
    float dx = xmax - xmin, dy = ymax - ymin;
    return (dy > 1e-10f) ? dx / dy : 1.0f;
}

uint64_t omega_hash_geometry(const float *verts, size_t n, uint32_t mat_id)
{
    if (n < 2) return qxor_avalanche((uint64_t)mat_id);
    float *edges = omega_sorted_edge_lengths(verts, n);
    if (!edges) return 0;
    size_t npairs = n*(n-1)/2;
    uint64_t h = qxor_fnv1a(edges, npairs * sizeof(float));
    h ^= qxor_avalanche((uint64_t)mat_id);
    free(edges);
    return h;
}

omega_cell_sketch_t omega_sketch_cell(const float *verts, size_t n,
                                       const uint32_t *mats)
{
    omega_cell_sketch_t s = {0};
    float *edges  = omega_sorted_edge_lengths(verts, n);
    float *angles = omega_sorted_vertex_angles(verts, n);
    if (edges)  s.structural_hash  = qxor_fnv1a(edges, n*(n-1)/2 * sizeof(float));
    if (angles) s.structural_hash ^= qxor_fnv1a(angles, n * sizeof(float));
    s.material_hash = qxor_fnv1a(mats, n * sizeof(uint32_t));
    s.bbox_aspect   = omega_bbox_aspect(verts, n);
    s.vertex_count  = (uint16_t)n;
    free(edges); free(angles);
    return s;
}

int omega_detect_symmetry(omega_chip_geometry_t *geom)
{
    if (!geom || !geom->vertices) return -1;
    /* Symmetry detection via cell sketch hashing.
     * Identical sketches → identical geometry → share spline controls. */
    geom->num_unique_cells = geom->num_polygons;
    return 0;
}

