#ifndef OMEGA_SITTON_H
#define OMEGA_SITTON_H
/*
 * omega_sitton.h — Sitton Number Arithmetic
 *
 * Extends ℝ with a nilpotent probability dimension 𝔭 where 𝔭² = 0, 𝔭 ≠ 0.
 *
 * Every Sitton number z = a + 𝔭·p carries both a classical value (a)
 * and an uncertainty amplitude (p). The nilpotent property gives
 * automatic first-order sensitivity analysis: no finite differences,
 * no perturbation series — the algebra does the calculus for you.
 *
 *   exp(a + 𝔭·p) = exp(a) + 𝔭·p·exp(a)   ← series terminates exactly
 *   (a + 𝔭·p)²  = a² + 2a·𝔭·p            ← derivative falls out free
 *
 * Named for the Cayley-Dickson extension with probability operator.
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 * SPDX-License-Identifier: Proprietary
 */

#include <stdint.h>
#include <stdbool.h>
#include <math.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ╔══════════════════════════════════════════════════════════════╗
 * ║  Core Type: sitton_f32_t — packed to 8 bytes for coalesced  ║
 * ║  GPU access and atomic 64-bit load/store on x86-64.         ║
 * ╚══════════════════════════════════════════════════════════════╝ */

typedef struct __attribute__((packed, aligned(8))) {
    float   real;   /* Classical component: the measured value       */
    float   prob;   /* Probability amplitude: the uncertainty field  */
} sitton_f32_t;

typedef struct __attribute__((packed, aligned(16))) {
    double  real;
    double  prob;
} sitton_f64_t;

/* ── Construction ────────────────────────────────────────────── */

static inline sitton_f32_t sn_make(float real, float prob)
{ return (sitton_f32_t){ .real = real, .prob = prob }; }

static inline sitton_f32_t sn_pure(float real)
{ return (sitton_f32_t){ .real = real, .prob = 0.0f }; }

static inline sitton_f32_t sn_zero(void)
{ return (sitton_f32_t){ .real = 0.0f, .prob = 0.0f }; }

static inline sitton_f32_t sn_one(void)
{ return (sitton_f32_t){ .real = 1.0f, .prob = 0.0f }; }

/* 𝔭 itself — the nilpotent unit */
static inline sitton_f32_t sn_pfrak(void)
{ return (sitton_f32_t){ .real = 0.0f, .prob = 1.0f }; }

/* ── Arithmetic ──────────────────────────────────────────────── */

/* (a₁+𝔭p₁) + (a₂+𝔭p₂) = (a₁+a₂) + 𝔭(p₁+p₂) */
static inline sitton_f32_t sn_add(sitton_f32_t x, sitton_f32_t y)
{ return (sitton_f32_t){ .real = x.real + y.real, .prob = x.prob + y.prob }; }

static inline sitton_f32_t sn_sub(sitton_f32_t x, sitton_f32_t y)
{ return (sitton_f32_t){ .real = x.real - y.real, .prob = x.prob - y.prob }; }

/*
 * (a₁+𝔭p₁)(a₂+𝔭p₂) = a₁a₂ + 𝔭(a₁p₂ + p₁a₂)
 *
 * The 𝔭²p₁p₂ term vanishes. This is not an approximation.
 * This is exact. The algebra itself performs the truncation.
 */
static inline sitton_f32_t sn_mul(sitton_f32_t x, sitton_f32_t y)
{
    return (sitton_f32_t){
        .real = x.real * y.real,
        .prob = x.real * y.prob + x.prob * y.real
    };
}

/*
 * Division: multiply by conjugate / norm².
 * (a₁+𝔭p₁)/(a₂+𝔭p₂) = (a₁/a₂) + 𝔭(p₁a₂ - a₁p₂)/a₂²
 */
static inline sitton_f32_t sn_div(sitton_f32_t x, sitton_f32_t y)
{
    if (__builtin_expect(fabsf(y.real) < FLT_MIN, 0))
        return (sitton_f32_t){ .real = __builtin_inff(), .prob = __builtin_inff() };
    float inv  = 1.0f / y.real;
    float inv2 = inv * inv;
    return (sitton_f32_t){
        .real = x.real * inv,
        .prob = (x.prob * y.real - x.real * y.prob) * inv2
    };
}

static inline sitton_f32_t sn_scale(sitton_f32_t x, float alpha)
{ return (sitton_f32_t){ .real = x.real * alpha, .prob = x.prob * alpha }; }

static inline sitton_f32_t sn_neg(sitton_f32_t x)
{ return (sitton_f32_t){ .real = -x.real, .prob = -x.prob }; }

/* conj(a+𝔭p) = a-𝔭p */
static inline sitton_f32_t sn_conj(sitton_f32_t x)
{ return (sitton_f32_t){ .real = x.real, .prob = -x.prob }; }

/* Fused multiply-add: x*y + z  (uses fmaf for real part) */
static inline sitton_f32_t sn_fma(sitton_f32_t x, sitton_f32_t y, sitton_f32_t z)
{
    return (sitton_f32_t){
        .real = fmaf(x.real, y.real, z.real),
        .prob = fmaf(x.real, y.prob, fmaf(x.prob, y.real, z.prob))
    };
}

/* ── Transcendentals (exact — series terminates) ─────────────── */

/* exp(a+𝔭p) = exp(a)(1 + 𝔭p) */
static inline sitton_f32_t sn_exp(sitton_f32_t x)
{ float ea = expf(x.real); return (sitton_f32_t){ .real = ea, .prob = x.prob * ea }; }

/* ln(a+𝔭p) = ln(a) + 𝔭·p/a */
static inline sitton_f32_t sn_log(sitton_f32_t x)
{
    if (x.real <= 0.0f) return (sitton_f32_t){ .real = __builtin_nanf(""), .prob = __builtin_nanf("") };
    return (sitton_f32_t){ .real = logf(x.real), .prob = x.prob / x.real };
}

/* sqrt(a+𝔭p) = sqrt(a) + 𝔭·p/(2·sqrt(a)) */
static inline sitton_f32_t sn_sqrt(sitton_f32_t x)
{
    if (x.real < 0.0f) return (sitton_f32_t){ .real = __builtin_nanf(""), .prob = __builtin_nanf("") };
    if (x.real < FLT_MIN) return (sitton_f32_t){ .real = 0.0f, .prob = __builtin_inff() };
    float sa = sqrtf(x.real);
    return (sitton_f32_t){ .real = sa, .prob = x.prob / (2.0f * sa) };
}

/* sin(a+𝔭p) = sin(a) + 𝔭·p·cos(a) */
static inline sitton_f32_t sn_sin(sitton_f32_t x)
{ return (sitton_f32_t){ .real = sinf(x.real), .prob = x.prob * cosf(x.real) }; }

/* cos(a+𝔭p) = cos(a) - 𝔭·p·sin(a) */
static inline sitton_f32_t sn_cos(sitton_f32_t x)
{ return (sitton_f32_t){ .real = cosf(x.real), .prob = -x.prob * sinf(x.real) }; }

/* ── Analysis ────────────────────────────────────────────────── */

/* Sensitivity: |prob/real| — fragility metric */
static inline float sn_sensitivity(sitton_f32_t x)
{ return (fabsf(x.real) > FLT_MIN) ? fabsf(x.prob / x.real) : __builtin_inff(); }

/* Dark mode: prob < 0 → destructive interference suppresses this state */
static inline bool sn_is_dark(sitton_f32_t x)
{ return x.prob < 0.0f; }

/* Hyper-certain: |prob| > 1 → resonant enhancement */
static inline bool sn_is_hyper(sitton_f32_t x)
{ return fabsf(x.prob) > 1.0f; }

/* Purely nilpotent: real=0, prob≠0 — the mathematical ghost */
static inline bool sn_is_nilpotent(sitton_f32_t x)
{ return fabsf(x.real) < FLT_MIN && fabsf(x.prob) >= FLT_MIN; }

/* ── Accumulator (running ensemble stats) ────────────────────── */

typedef struct {
    sitton_f32_t    sum;
    sitton_f32_t    sum_sq;
    float           worst_sensitivity;
    float           best_sensitivity;
    uint32_t        count;
    uint32_t        dark_count;
} sn_accumulator_t;

void sn_accum_init(sn_accumulator_t *acc);
void sn_accum_push(sn_accumulator_t *acc, sitton_f32_t value);
sitton_f32_t sn_accum_mean(const sn_accumulator_t *acc);
sitton_f32_t sn_accum_variance(const sn_accumulator_t *acc);
float sn_accum_dark_fraction(const sn_accumulator_t *acc);

/* ── Coupling type (shared with QXOR and VRAM) ──────────────── */

typedef struct {
    uint32_t        qubit_a, qubit_b;
    sitton_f32_t    coupling_g;
    sitton_f32_t    coupling_phase;
    sitton_f32_t    frequency_shift;
} omega_sitton_coupling_t;

/* ── 64-bit variants ─────────────────────────────────────────── */

static inline sitton_f64_t sn64_make(double real, double prob)
{ return (sitton_f64_t){ .real = real, .prob = prob }; }

static inline sitton_f64_t sn64_mul(sitton_f64_t x, sitton_f64_t y)
{ return (sitton_f64_t){ .real = x.real * y.real, .prob = x.real * y.prob + x.prob * y.real }; }

static inline sitton_f64_t sn64_exp(sitton_f64_t x)
{ double ea = exp(x.real); return (sitton_f64_t){ .real = ea, .prob = x.prob * ea }; }

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_SITTON_H */

