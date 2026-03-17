/*
 * omega_sitton.c — Sitton Number Accumulator Implementation
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_sitton.h"
#include <string.h>

void sn_accum_init(sn_accumulator_t *acc)
{
    memset(acc, 0, sizeof(*acc));
    acc->best_sensitivity  = __builtin_inff();
    acc->worst_sensitivity = 0.0f;
}

void sn_accum_push(sn_accumulator_t *acc, sitton_f32_t value)
{
    acc->sum    = sn_add(acc->sum, value);
    acc->sum_sq = sn_add(acc->sum_sq, sn_mul(value, value));
    acc->count++;

    float sens = sn_sensitivity(value);
    if (sens > acc->worst_sensitivity) acc->worst_sensitivity = sens;
    if (sens < acc->best_sensitivity)  acc->best_sensitivity  = sens;
    if (sn_is_dark(value)) acc->dark_count++;
}

sitton_f32_t sn_accum_mean(const sn_accumulator_t *acc)
{
    if (acc->count == 0) return sn_zero();
    return sn_scale(acc->sum, 1.0f / (float)acc->count);
}

sitton_f32_t sn_accum_variance(const sn_accumulator_t *acc)
{
    if (acc->count < 2) return sn_zero();
    float inv_n = 1.0f / (float)acc->count;
    sitton_f32_t mean    = sn_scale(acc->sum, inv_n);
    sitton_f32_t mean_sq = sn_scale(acc->sum_sq, inv_n);
    return sn_sub(mean_sq, sn_mul(mean, mean));
}

float sn_accum_dark_fraction(const sn_accumulator_t *acc)
{
    if (acc->count == 0) return 0.0f;
    return (float)acc->dark_count / (float)acc->count;
}

