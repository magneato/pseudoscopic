/*
 * omega_example.c — ΩMEGA Demonstration
 *
 * Demonstrates the complete ΩMEGA pipeline:
 *   1. Sitton Number arithmetic
 *   2. QXOR path signatures
 *   3. Neural Spline hierarchy
 *   4. TruRNG3 entropy
 *   5. Hashbelt caching
 *   6. Full chip simulation
 *
 * Build: gcc -O2 -I../src -o omega_example omega_example.c \
 *        -L.. -lnearmem -lpthread -lm
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <stdio.h>
#include <math.h>
#include "omega_sitton.h"
#include "omega_qxor.h"
#include "omega_entropy.h"
#include "omega_hashbelt.h"

static void demo_sitton(void)
{
    printf("=== Sitton Number Arithmetic ===\n\n");

    sitton_f32_t x = sn_make(3.0f, 0.1f);   /* 3.0 + pfrak*0.1  */
    sitton_f32_t y = sn_make(2.0f, -0.05f);  /* 2.0 - pfrak*0.05 */

    sitton_f32_t prod = sn_mul(x, y);
    printf("  x       = %.4f + pfrak*%.4f\n", x.real, x.prob);
    printf("  y       = %.4f + pfrak*%.4f\n", y.real, y.prob);
    printf("  x*y     = %.4f + pfrak*%.4f\n", prod.real, prod.prob);
    printf("  exact:    6.0000 + pfrak*0.0500\n");
    printf("  verify:   3*2=6, 3*(-0.05)+0.1*2=0.05  [pfrak^2=0]\n\n");

    sitton_f32_t ex = sn_exp(sn_make(1.0f, 0.01f));
    printf("  exp(1 + pfrak*0.01) = %.6f + pfrak*%.6f\n", ex.real, ex.prob);
    printf("  exact:  e*(1 + pfrak*0.01) = %.6f + pfrak*%.6f\n\n",
           expf(1.0f), 0.01f * expf(1.0f));

    sitton_f32_t dark = sn_make(0.5f, -0.3f);
    printf("  dark mode test: %.2f + pfrak*(%.2f) -> is_dark=%d\n",
           dark.real, dark.prob, sn_is_dark(dark));
    printf("  sensitivity: %.4f (%.4f/%.4f)\n\n",
           sn_sensitivity(dark), fabsf(dark.prob), dark.real);
}

static void demo_qxor(void)
{
    printf("=== QXOR Path Signatures ===\n\n");

    qxor_path_t p1 = qxor_path_begin();
    qxor_path_t p2 = qxor_path_begin();

    /* Two paths through identical geometry → identical signatures */
    uint64_t seg_a = 0xCAFEBABE12345678ULL;
    uint64_t seg_b = 0xDEADBEEF87654321ULL;

    qxor_path_extend(&p1, seg_a, 50.0f);
    qxor_path_extend(&p1, seg_b, 30.0f);

    qxor_path_extend(&p2, seg_a, 50.0f);
    qxor_path_extend(&p2, seg_b, 30.0f);

    printf("  Path 1 sig: 0x%016lx (%.0f um)\n", p1.signature, p1.path_length_um);
    printf("  Path 2 sig: 0x%016lx (%.0f um)\n", p2.signature, p2.path_length_um);
    printf("  Equivalent: %s\n", qxor_paths_equivalent(p1, p2, 0) ? "YES" : "no");
    printf("  Hamming:    %lu\n\n", qxor_hamming(p1.signature, p2.signature));

    /* Bit hack showcase */
    printf("  popcount(0xFF00FF00FF00FF00) = %lu\n", qxor_popcount64(0xFF00FF00FF00FF00ULL));
    printf("  next_pow2(1000) = %lu\n", qxor_next_pow2(1000));
    printf("  log2(1024) = %u\n", qxor_log2(1024));
    printf("  morton2d(3,5) = 0x%016lx\n\n", qxor_morton2d(3, 5));
}

static void demo_hashbelt(void)
{
    printf("=== Lock-Free Hashbelt ===\n\n");

    hb_belt_t belt;
    hb_belt_init(&belt, 256, 64, 4096);

    /* Insert some cached path signatures */
    for (uint64_t i = 0; i < 100; i++)
        hb_belt_insert(&belt, i * 0x100, i * 42);

    bool found;
    uint64_t val = hb_belt_lookup(&belt, 50 * 0x100, &found);
    printf("  Lookup key 0x%X: %s, value=%lu\n", 50 * 0x100, found ? "HIT" : "miss", val);

    /* Advance belt to evict old entries */
    uint32_t evicted = hb_belt_advance(&belt);
    printf("  Belt advance: evicted %u entries\n", evicted);

    uint64_t total_ins = atomic_load(&belt.total_inserts);
    uint64_t total_look = atomic_load(&belt.total_lookups);
    uint64_t total_hits = atomic_load(&belt.total_hits);
    printf("  Stats: inserts=%lu lookups=%lu hits=%lu\n\n",
           total_ins, total_look, total_hits);

    hb_belt_destroy(&belt);
}

int main(void)
{
    printf("\n  OMEGA Example — Quantum Chip EM Simulation Toolkit\n");
    printf("  (c) 2026 Neural Splines LLC\n\n");

    demo_sitton();
    demo_qxor();
    demo_hashbelt();

    printf("=== All demos complete ===\n\n");
    return 0;
}

