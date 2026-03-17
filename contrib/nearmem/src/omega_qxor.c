/*
 * omega_qxor.c — QXOR Relational Encoding Implementation
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_qxor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Weak XOR buffers: overridden by asm/xor_cascade.asm ─────── */

__attribute__((weak))
void qxor_xor_buffers(void *dst, const void *src, size_t size)
{
    uint64_t       *d = (uint64_t *)dst;
    const uint64_t *s = (const uint64_t *)src;
    size_t n = size >> 3;
    for (size_t i = 0; i < n; i++) d[i] ^= s[i];
    uint8_t       *dt = (uint8_t *)(d + n);
    const uint8_t *st = (const uint8_t *)(s + n);
    for (size_t i = 0; i < (size & 7); i++) dt[i] ^= st[i];
}

void qxor_xor_with_prng(void *buf, size_t size, uint64_t seed)
{
    qxor_rng_t rng;
    qxor_rng_seed(&rng, seed);
    uint64_t *d = (uint64_t *)buf;
    size_t n = size >> 3;
    for (size_t i = 0; i < n; i++) d[i] ^= qxor_rng_next(&rng);
    if (size & 7) {
        uint64_t tail = qxor_rng_next(&rng);
        uint8_t *dt = (uint8_t *)(d + n);
        for (size_t i = 0; i < (size & 7); i++) dt[i] ^= ((uint8_t *)&tail)[i];
    }
}

uint64_t qxor_checksum(const void *data, size_t size)
{
    const uint64_t *d = (const uint64_t *)data;
    uint64_t csum = 0;
    for (size_t i = 0; i < (size >> 3); i++) csum ^= d[i];
    if (size & 7) {
        uint64_t tail = 0;
        memcpy(&tail, d + (size >> 3), size & 7);
        csum ^= tail;
    }
    return csum;
}

/* ── Compressibility (Shannon entropy / 8) ───────────────────── */

float omega_xor_compressibility(const void *data, size_t size)
{
    if (size == 0) return 1.0f;
    uint64_t hist[256] = {0};
    const uint8_t *p = (const uint8_t *)data;
    for (size_t i = 0; i < size; i++) hist[p[i]]++;
    double H = 0.0, inv = 1.0 / (double)size;
    for (int i = 0; i < 256; i++) {
        if (hist[i] == 0) continue;
        double f = (double)hist[i] * inv;
        H -= f * log2(f);
    }
    return (float)(1.0 - H / 8.0);
}

/* ── XOR Cascade ─────────────────────────────────────────────── */

int omega_xor_cascade_compress(const void *input, size_t input_size,
                               uint64_t master_seed, uint8_t max_levels,
                               omega_xor_cascade_t *cascade,
                               uint8_t *work_buf, size_t buf_cap)
{
    if (!input || !cascade || !work_buf || buf_cap < input_size) return -1;
    if (max_levels > OMEGA_MAX_XOR_LEVELS) max_levels = OMEGA_MAX_XOR_LEVELS;

    memcpy(work_buf, input, input_size);
    cascade->original_size = input_size;
    cascade->num_levels = 0;

    qxor_rng_t seed_gen;
    qxor_rng_seed(&seed_gen, master_seed);
    float best = omega_xor_compressibility(work_buf, input_size);

    for (uint8_t lvl = 0; lvl < max_levels; lvl++) {
        uint64_t lseed = qxor_rng_next(&seed_gen);
        cascade->seeds[lvl] = lseed;
        qxor_xor_with_prng(work_buf, input_size, lseed);
        cascade->num_levels = lvl + 1;

        float c = omega_xor_compressibility(work_buf, input_size);
        if (c <= best + 0.01f) {
            /* Undo and stop — this level didn't help */
            qxor_xor_with_prng(work_buf, input_size, lseed);
            cascade->num_levels = lvl;
            break;
        }
        best = c;
    }
    cascade->compressed_data = work_buf;
    cascade->compressed_size = input_size;
    return 0;
}

int omega_xor_cascade_decompress(const omega_xor_cascade_t *cascade,
                                 void *output, size_t output_size)
{
    if (!cascade || !output || output_size < cascade->original_size) return -1;
    memcpy(output, cascade->compressed_data, cascade->compressed_size);
    for (int lvl = (int)cascade->num_levels - 1; lvl >= 0; lvl--)
        qxor_xor_with_prng(output, cascade->original_size, cascade->seeds[lvl]);
    return 0;
}

/* ── Coupling Graph ──────────────────────────────────────────── */

int omega_graph_init(omega_coupling_graph_t *g, size_t nq, size_t cap)
{
    memset(g, 0, sizeof(*g));
    g->couplings = (omega_coupling_t *)calloc(cap, sizeof(omega_coupling_t));
    if (!g->couplings) return -1;
    g->capacity = cap;
    g->num_qubits = nq;
    return 0;
}

int omega_graph_add_edge(omega_coupling_graph_t *g, const omega_coupling_t *e)
{
    if (g->num_couplings >= g->capacity) {
        size_t nc = g->capacity ? g->capacity * 2 : 64;
        omega_coupling_t *p = (omega_coupling_t *)realloc(g->couplings, nc * sizeof(*p));
        if (!p) return -1;
        g->couplings = p;
        g->capacity = nc;
    }
    g->couplings[g->num_couplings++] = *e;
    g->csr_valid = false;
    return 0;
}

static int edge_cmp_fn(const void *a, const void *b) {
    const omega_coupling_t *ea = (const omega_coupling_t *)a;
    const omega_coupling_t *eb = (const omega_coupling_t *)b;
    if (ea->qubit_a != eb->qubit_a) return (ea->qubit_a < eb->qubit_a) ? -1 : 1;
    if (ea->qubit_b != eb->qubit_b) return (ea->qubit_b < eb->qubit_b) ? -1 : 1;
    return 0;
}

int omega_graph_finalize(omega_coupling_graph_t *g)
{
    if (g->num_couplings > 1)
        qsort(g->couplings, g->num_couplings, sizeof(omega_coupling_t), edge_cmp_fn);

    free(g->row_offsets); free(g->col_indices); free(g->weights);
    uint32_t nq = (uint32_t)g->num_qubits;
    g->row_offsets = (uint32_t *)calloc(nq + 1, sizeof(uint32_t));
    g->col_indices = (uint32_t *)calloc(g->num_couplings, sizeof(uint32_t));
    g->weights     = (float *)calloc(g->num_couplings, sizeof(float));
    if (!g->row_offsets || !g->col_indices || !g->weights) return -1;

    for (size_t i = 0; i < g->num_couplings; i++) {
        uint32_t qa = g->couplings[i].qubit_a;
        if (qa < nq) g->row_offsets[qa + 1]++;
    }
    for (uint32_t i = 1; i <= nq; i++) g->row_offsets[i] += g->row_offsets[i-1];

    uint32_t *cur = (uint32_t *)calloc(nq, sizeof(uint32_t));
    if (!cur) return -1;
    for (size_t i = 0; i < g->num_couplings; i++) {
        uint32_t qa = g->couplings[i].qubit_a;
        if (qa < nq) {
            uint32_t pos = g->row_offsets[qa] + cur[qa]++;
            g->col_indices[pos] = g->couplings[i].qubit_b;
            g->weights[pos]     = g->couplings[i].coupling_g;
        }
    }
    free(cur);
    g->csr_valid = true;
    return 0;
}

void omega_graph_destroy(omega_coupling_graph_t *g)
{
    free(g->couplings); free(g->row_offsets);
    free(g->col_indices); free(g->weights);
    memset(g, 0, sizeof(*g));
}

size_t omega_graph_find_equivalent(const omega_coupling_graph_t *g,
                                   uint64_t sig, unsigned tol,
                                   uint32_t *out, size_t max)
{
    size_t found = 0;
    for (size_t i = 0; i < g->num_couplings && found < max; i++)
        if (qxor_hamming(g->couplings[i].path_signature, sig) <= tol)
            out[found++] = (uint32_t)i;
    return found;
}

int omega_build_coupling_graph(const omega_chip_geometry_t *geom,
                               const omega_spline_field_t *field,
                               omega_coupling_graph_t *graph)
{
    (void)field;
    if (!geom || !graph) return -1;

    /* Build path signatures between all qubit pairs via geometry hashing */
    for (size_t i = 0; i < graph->num_qubits; i++) {
        for (size_t j = i + 1; j < graph->num_qubits; j++) {
            qxor_path_t path = qxor_path_begin();
            /* Hash the geometry between qubits i and j.
             * In production: trace through actual chip topology.
             * Here: use material hashes as proxy. */
            uint64_t seg_hash = qxor_avalanche(
                qxor_fnv1a(&geom->materials[0], sizeof(omega_material_t)));
            qxor_path_extend(&path, seg_hash, 100.0f);

            omega_coupling_t edge = {
                .qubit_a       = (uint32_t)i,
                .qubit_b       = (uint32_t)j,
                .coupling_g    = 0.0f,
                .coupling_phase = 0.0f,
                .distance_um   = path.path_length_um,
                .path_signature = path.signature
            };
            omega_graph_add_edge(graph, &edge);
        }
    }
    return omega_graph_finalize(graph);
}

