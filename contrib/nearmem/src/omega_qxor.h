#ifndef OMEGA_QXOR_H
#define OMEGA_QXOR_H
/*
 * omega_qxor.h — QXOR Relational Encoding via XOR Path Signatures
 *
 * XOR is its own inverse: a ⊕ b ⊕ b = a
 * This makes it the natural operation for bidirectional relationships.
 *
 * Instead of propagating waves through 11 billion cells to discover
 * coupling, we XOR geometry along coupling paths.
 * Identical signatures ≡ identical coupling. d.a.t.a. principle:
 * same data = same architecture = same answer.
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "omega_geometry.h"
#include "omega_spline.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ── Bit Hacks — branchless, compiler-intrinsic where possible ── */

static inline uint64_t qxor_popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return (uint64_t)__builtin_popcountll(x);
#else
    x -= (x >> 1) & 0x5555555555555555ULL;
    x  = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x  = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (x * 0x0101010101010101ULL) >> 56;
#endif
}

/* Hamming distance: the QXOR metric */
static inline uint64_t qxor_hamming(uint64_t a, uint64_t b)
{ return qxor_popcount64(a ^ b); }

/* XOR-rotate: mix bits non-linearly without multiplication */
static inline uint64_t qxor_mix(uint64_t x, unsigned shift)
{ return x ^ ((x << shift) | (x >> (64 - shift))); }

/* Parity: branchless, single instruction on x86 */
static inline uint8_t qxor_parity64(uint64_t x)
{ return (uint8_t)(qxor_popcount64(x) & 1); }

/* Next power of two. 0 → 0. */
static inline uint64_t qxor_next_pow2(uint64_t x) {
    if (x == 0) return 0;
    x--; x |= x >> 1; x |= x >> 2; x |= x >> 4;
    x |= x >> 8; x |= x >> 16; x |= x >> 32;
    return x + 1;
}

/* Fast integer log2 (floor). Undefined for x==0. */
static inline unsigned qxor_log2(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return 63 - (unsigned)__builtin_clzll(x);
#else
    unsigned r = 0; while (x >>= 1) r++; return r;
#endif
}

/* Byte-swap (endian conversion) */
static inline uint64_t qxor_bswap64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return __builtin_bswap64(x);
#else
    return ((x & 0x00000000000000FFULL) << 56) |
           ((x & 0x000000000000FF00ULL) << 40) |
           ((x & 0x0000000000FF0000ULL) << 24) |
           ((x & 0x00000000FF000000ULL) <<  8) |
           ((x & 0x000000FF00000000ULL) >>  8) |
           ((x & 0x0000FF0000000000ULL) >> 24) |
           ((x & 0x00FF000000000000ULL) >> 40) |
           ((x & 0xFF00000000000000ULL) >> 56);
#endif
}

/* Morton interleave for 2D spatial hashing */
static inline uint64_t qxor_morton2d(uint32_t a, uint32_t b) {
    uint64_t x = a, y = b;
    x = (x | (x << 16)) & 0x0000FFFF0000FFFFULL;
    x = (x | (x <<  8)) & 0x00FF00FF00FF00FFULL;
    x = (x | (x <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
    x = (x | (x <<  2)) & 0x3333333333333333ULL;
    x = (x | (x <<  1)) & 0x5555555555555555ULL;
    y = (y | (y << 16)) & 0x0000FFFF0000FFFFULL;
    y = (y | (y <<  8)) & 0x00FF00FF00FF00FFULL;
    y = (y | (y <<  4)) & 0x0F0F0F0F0F0F0F0FULL;
    y = (y | (y <<  2)) & 0x3333333333333333ULL;
    y = (y | (y <<  1)) & 0x5555555555555555ULL;
    return x | (y << 1);
}

/* ── SplitMix64 PRNG — cascade seed expander ────────────────── */

typedef struct { uint64_t state; } qxor_rng_t;

static inline void qxor_rng_seed(qxor_rng_t *rng, uint64_t seed)
{ rng->state = seed; }

static inline uint64_t qxor_rng_next(qxor_rng_t *rng) {
    uint64_t z = (rng->state += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

static inline float qxor_rng_float(qxor_rng_t *rng)
{ return (float)(qxor_rng_next(rng) >> 40) * 0x1.0p-24f; }

/* ── FNV-1a 64-bit Hash ─────────────────────────────────────── */

#define QXOR_FNV_OFFSET  0xCBF29CE484222325ULL
#define QXOR_FNV_PRIME   0x00000100000001B3ULL

static inline uint64_t qxor_fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint64_t h = QXOR_FNV_OFFSET;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= QXOR_FNV_PRIME; }
    return h;
}

/* Avalanche finalizer (Stafford variant 13 of Murmur3) */
static inline uint64_t qxor_avalanche(uint64_t x) {
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ULL;
    x ^= x >> 27; x *= 0x94D049BB133111EBULL;
    x ^= x >> 31; return x;
}

/* ── Path Signatures ─────────────────────────────────────────── */

typedef struct {
    uint64_t    signature;
    uint32_t    segment_count;
    float       path_length_um;
} qxor_path_t;

static inline qxor_path_t qxor_path_begin(void)
{ return (qxor_path_t){ .signature = 0, .segment_count = 0, .path_length_um = 0.0f }; }

static inline void qxor_path_extend(qxor_path_t *path,
                                     uint64_t seg_hash, float seg_length)
{
    unsigned rot = (path->segment_count % 63) + 1;
    path->signature ^= qxor_mix(seg_hash, rot);
    path->segment_count++;
    path->path_length_um += seg_length;
}

static inline bool qxor_paths_equivalent(qxor_path_t a, qxor_path_t b,
                                          unsigned hamming_tol)
{ return qxor_hamming(a.signature, b.signature) <= hamming_tol; }

/* ── XOR Cascade ─────────────────────────────────────────────── */

#define OMEGA_MAX_XOR_LEVELS 10

typedef struct {
    uint64_t    seeds[OMEGA_MAX_XOR_LEVELS];
    uint8_t     num_levels;
    size_t      original_size;
    size_t      compressed_size;
    uint8_t     *compressed_data;
} omega_xor_cascade_t;

int  omega_xor_cascade_compress(const void *input, size_t input_size,
                                uint64_t master_seed, uint8_t max_levels,
                                omega_xor_cascade_t *cascade,
                                uint8_t *work_buf, size_t buf_cap);

int  omega_xor_cascade_decompress(const omega_xor_cascade_t *cascade,
                                  void *output, size_t output_size);

float omega_xor_compressibility(const void *data, size_t size);

/* ── Coupling Graph (CSR sparse format) ──────────────────────── */

typedef struct {
    uint32_t    qubit_a;
    uint32_t    qubit_b;
    float       coupling_g;
    float       coupling_phase;
    float       distance_um;
    uint64_t    path_signature;
} omega_coupling_t;

typedef struct {
    omega_coupling_t *couplings;
    size_t      num_couplings;
    size_t      capacity;
    size_t      num_qubits;
    uint32_t   *row_offsets;     /* CSR: [num_qubits+1] prefix sums  */
    uint32_t   *col_indices;     /* CSR: [num_couplings]              */
    float      *weights;         /* CSR: [num_couplings]              */
    bool        csr_valid;
} omega_coupling_graph_t;

int    omega_graph_init(omega_coupling_graph_t *g, size_t nq, size_t cap);
int    omega_graph_add_edge(omega_coupling_graph_t *g, const omega_coupling_t *e);
int    omega_graph_finalize(omega_coupling_graph_t *g);
void   omega_graph_destroy(omega_coupling_graph_t *g);
size_t omega_graph_find_equivalent(const omega_coupling_graph_t *g,
                                   uint64_t sig, unsigned tol,
                                   uint32_t *out, size_t max);

/* bulk XOR operations — weak symbol, overridden by NASM */
void qxor_xor_buffers(void *dst, const void *src, size_t size);
void qxor_xor_with_prng(void *buf, size_t size, uint64_t seed);
uint64_t qxor_checksum(const void *data, size_t size);

int omega_build_coupling_graph(const omega_chip_geometry_t *geom,
                               const omega_spline_field_t *field,
                               omega_coupling_graph_t *graph);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_QXOR_H */

