#ifndef OMEGA_GEOMETRY_H
#define OMEGA_GEOMETRY_H
/*
 * omega_geometry.h — Chip Geometry with Symmetry Detection
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float   epsilon_r;
    float   mu_r;
    float   sigma;
    float   tan_delta;
    float   thickness_um;
    uint8_t is_superconductor;
    float   Tc_kelvin;
    float   london_depth_nm;
    float   fab_tolerance_nm;
} omega_material_t;

typedef struct {
    uint64_t    structural_hash;
    uint64_t    material_hash;
    float       bbox_aspect;
    uint16_t    vertex_count;
} omega_cell_sketch_t;

typedef struct {
    uint32_t placeholder;
} omega_sym_group_t;

typedef struct {
    uint32_t placeholder;
} omega_cell_t;

typedef struct {
    float              *vertices;
    uint32_t           *material_ids;
    size_t              num_polygons;
    omega_sym_group_t   symmetry;
    omega_cell_t       *unit_cell;
    uint32_t            num_repeats;
    float              *offsets;
    omega_material_t   *materials;
    size_t              num_materials;
    size_t              num_unique_cells;
    uint16_t           *cell_to_unique;
} omega_chip_geometry_t;

omega_cell_sketch_t omega_sketch_cell(const float *verts, size_t n,
                                       const uint32_t *mats);
uint64_t omega_hash_geometry(const float *verts, size_t n, uint32_t mat_id);
float   *omega_sorted_edge_lengths(const float *verts, size_t n);
float   *omega_sorted_vertex_angles(const float *verts, size_t n);
float    omega_bbox_aspect(const float *verts, size_t n);
int      omega_detect_symmetry(omega_chip_geometry_t *geom);

/* Standard materials for superconducting quantum chips */
extern const omega_material_t OMEGA_MAT_NIOBIUM;
extern const omega_material_t OMEGA_MAT_ALUMINUM;
extern const omega_material_t OMEGA_MAT_SILICON;
extern const omega_material_t OMEGA_MAT_SAPPHIRE;
extern const omega_material_t OMEGA_MAT_VACUUM;

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_GEOMETRY_H */

