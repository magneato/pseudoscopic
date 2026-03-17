/*
 * omega_spline.c — Neural Spline™ Field Compression Implementation
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  PROPRIETARY: Neural Splines™ — Neural Splines LLC           ║
 * ║  (c) 2026 Robert L. Sitton, Jr. — All Rights Reserved       ║
 * ╚═══════════════════════════════════════════════════════════════╝
 */

#include "omega_spline.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static inline float clampf(float x, float lo, float hi)
{ return fminf(fmaxf(x, lo), hi); }

static inline size_t clampsz(size_t x, size_t lo, size_t hi)
{ return (x < lo) ? lo : (x > hi) ? hi : x; }

/* ── 3D Spline ───────────────────────────────────────────────── */

int ns_spline_3d_init(ns_spline_3d_t *s, size_t nx, size_t ny, size_t nz,
                      float xn, float xx, float yn, float yx, float zn, float zx)
{
    if (nx < 4 || ny < 4 || nz < 4) return -1;
    memset(s, 0, sizeof(*s));
    s->ctrl = (float *)calloc(nx * ny * nz, sizeof(float));
    if (!s->ctrl) return -1;
    s->nx = nx; s->ny = ny; s->nz = nz;
    s->x_min = xn; s->x_max = xx; s->y_min = yn; s->y_max = yx;
    s->z_min = zn; s->z_max = zx;
    s->inv_dx = (float)(nx - 3) / (xx - xn);
    s->inv_dy = (float)(ny - 3) / (yx - yn);
    s->inv_dz = (float)(nz - 3) / (zx - zn);
    return 0;
}

void ns_spline_3d_destroy(ns_spline_3d_t *s)
{ free(s->ctrl); memset(s, 0, sizeof(*s)); }

static inline float cp3(const ns_spline_3d_t *s, size_t ix, size_t iy, size_t iz) {
    ix = clampsz(ix, 0, s->nx - 1);
    iy = clampsz(iy, 0, s->ny - 1);
    iz = clampsz(iz, 0, s->nz - 1);
    return s->ctrl[(iz * s->ny + iy) * s->nx + ix];
}

float ns_spline_3d_eval(const ns_spline_3d_t *s, float x, float y, float z)
{
    float ux = (clampf(x, s->x_min, s->x_max) - s->x_min) * s->inv_dx;
    float uy = (clampf(y, s->y_min, s->y_max) - s->y_min) * s->inv_dy;
    float uz = (clampf(z, s->z_min, s->z_max) - s->z_min) * s->inv_dz;
    size_t sx = clampsz((size_t)ux, 0, s->nx - 4);
    size_t sy = clampsz((size_t)uy, 0, s->ny - 4);
    size_t sz = clampsz((size_t)uz, 0, s->nz - 4);
    float bx[4], by[4], bz[4];
    ns_basis_cubic(ux - (float)sx, bx);
    ns_basis_cubic(uy - (float)sy, by);
    ns_basis_cubic(uz - (float)sz, bz);

    float result = 0.0f;
    for (int dz = 0; dz < 4; dz++)
        for (int dy = 0; dy < 4; dy++) {
            float row = 0.0f;
            for (int dx = 0; dx < 4; dx++)
                row += bx[dx] * cp3(s, sx+dx, sy+dy, sz+dz);
            result += bz[dz] * by[dy] * row;
        }
    return result;
}

float ns_spline_3d_eval_grad(const ns_spline_3d_t *s, float x, float y, float z, float g[3])
{
    float ux = (clampf(x, s->x_min, s->x_max) - s->x_min) * s->inv_dx;
    float uy = (clampf(y, s->y_min, s->y_max) - s->y_min) * s->inv_dy;
    float uz = (clampf(z, s->z_min, s->z_max) - s->z_min) * s->inv_dz;
    size_t sx = clampsz((size_t)ux, 0, s->nx - 4);
    size_t sy = clampsz((size_t)uy, 0, s->ny - 4);
    size_t sz = clampsz((size_t)uz, 0, s->nz - 4);
    float bx[4], by[4], bz[4], dbx[4], dby[4], dbz[4];
    ns_basis_cubic(ux-(float)sx, bx); ns_basis_cubic_deriv(ux-(float)sx, dbx);
    ns_basis_cubic(uy-(float)sy, by); ns_basis_cubic_deriv(uy-(float)sy, dby);
    ns_basis_cubic(uz-(float)sz, bz); ns_basis_cubic_deriv(uz-(float)sz, dbz);

    float val = 0.0f; g[0] = g[1] = g[2] = 0.0f;
    for (int dz = 0; dz < 4; dz++)
        for (int dy = 0; dy < 4; dy++) {
            float wx = 0.0f, dwx = 0.0f;
            for (int dx = 0; dx < 4; dx++) {
                float c = cp3(s, sx+dx, sy+dy, sz+dz);
                wx  += bx[dx]  * c;
                dwx += dbx[dx] * c;
            }
            float byz = by[dy]*bz[dz], dbyz = dby[dy]*bz[dz], bydz = by[dy]*dbz[dz];
            val   += byz  * wx;
            g[0]  += byz  * dwx;
            g[1]  += dbyz * wx;
            g[2]  += bydz * wx;
        }
    g[0] *= s->inv_dx; g[1] *= s->inv_dy; g[2] *= s->inv_dz;
    return val;
}

sitton_f32_t ns_spline_3d_eval_sitton(const ns_spline_3d_t *s,
    sitton_f32_t x, sitton_f32_t y, sitton_f32_t z)
{
    float g[3];
    float v = ns_spline_3d_eval_grad(s, x.real, y.real, z.real, g);
    return sn_make(v, g[0]*x.prob + g[1]*y.prob + g[2]*z.prob);
}

/* ── Hierarchy Fitting ───────────────────────────────────────── */

static int fit_level(ns_spline_3d_t *sp, const float *data, size_t w, size_t h, size_t d)
{
    size_t total = sp->nx * sp->ny * sp->nz;
    float *wts = (float *)calloc(total, sizeof(float));
    if (!wts) return -1;
    memset(sp->ctrl, 0, total * sizeof(float));

    for (size_t iz = 0; iz < d; iz++) {
      float z = sp->z_min + (d>1 ? (float)iz/(float)(d-1) : 0.0f) * (sp->z_max - sp->z_min);
      float uz = (z - sp->z_min) * sp->inv_dz;
      size_t sz = clampsz((size_t)uz, 0, sp->nz-4);
      float bz[4]; ns_basis_cubic(uz-(float)sz, bz);
      for (size_t iy = 0; iy < h; iy++) {
        float y = sp->y_min + (h>1 ? (float)iy/(float)(h-1) : 0.0f) * (sp->y_max - sp->y_min);
        float uy = (y - sp->y_min) * sp->inv_dy;
        size_t sy = clampsz((size_t)uy, 0, sp->ny-4);
        float by[4]; ns_basis_cubic(uy-(float)sy, by);
        for (size_t ix = 0; ix < w; ix++) {
          float x = sp->x_min + (w>1 ? (float)ix/(float)(w-1) : 0.0f) * (sp->x_max - sp->x_min);
          float ux = (x - sp->x_min) * sp->inv_dx;
          size_t sx = clampsz((size_t)ux, 0, sp->nx-4);
          float bx[4]; ns_basis_cubic(ux-(float)sx, bx);
          float val = data[(iz*h+iy)*w+ix];
          for (int dz=0; dz<4; dz++)
            for (int dy=0; dy<4; dy++)
              for (int dx=0; dx<4; dx++) {
                size_t ci = ((sz+dz)*sp->ny+(sy+dy))*sp->nx+(sx+dx);
                float bw = bx[dx]*by[dy]*bz[dz];
                sp->ctrl[ci] += bw * val;
                wts[ci] += bw;
              }
        }
      }
    }
    for (size_t i = 0; i < total; i++)
        if (wts[i] > 1e-10f) sp->ctrl[i] /= wts[i];
    free(wts);
    return 0;
}

int ns_hierarchy_fit(ns_hierarchy_t *h, const float *data,
                     size_t w, size_t ht, size_t d,
                     float xn, float xx, float yn, float yx, float zn, float zx,
                     float target_rms, uint8_t max_levels, size_t base_ctrl)
{
    memset(h, 0, sizeof(*h));
    h->original_bytes = w * ht * d * sizeof(float);
    if (max_levels > NS_MAX_LEVELS) max_levels = NS_MAX_LEVELS;

    size_t total = w * ht * d;
    float *residual = (float *)malloc(total * sizeof(float));
    if (!residual) return -1;
    memcpy(residual, data, total * sizeof(float));

    for (uint8_t lvl = 0; lvl < max_levels; lvl++) {
        size_t nc = (base_ctrl < 4 ? 4 : base_ctrl) * (1u << lvl);
        if (nc < 4) nc = 4;
        if (ns_spline_3d_init(&h->levels[lvl], nc, nc, nc, xn, xx, yn, yx, zn, zx) != 0)
            { free(residual); return -1; }
        if (fit_level(&h->levels[lvl], residual, w, ht, d) != 0)
            { free(residual); return -1; }
        h->valid[lvl] = true;
        h->num_levels = lvl + 1;
        h->compressed_bytes += nc * nc * nc * sizeof(float);

        double ssq = 0.0;
        for (size_t iz = 0; iz < d; iz++) {
          float z = zn + (d>1 ? (float)iz/(float)(d-1) : 0) * (zx - zn);
          for (size_t iy = 0; iy < ht; iy++) {
            float y = yn + (ht>1 ? (float)iy/(float)(ht-1) : 0) * (yx - yn);
            for (size_t ix = 0; ix < w; ix++) {
              float x = xn + (w>1 ? (float)ix/(float)(w-1) : 0) * (xx - xn);
              size_t idx = (iz*ht+iy)*w+ix;
              residual[idx] -= ns_spline_3d_eval(&h->levels[lvl], x, y, z);
              ssq += (double)residual[idx] * (double)residual[idx];
            }
          }
        }
        h->rms_residual = (float)sqrt(ssq / (double)total);
        if (h->rms_residual <= target_rms) break;
    }

    float mx = 0.0f;
    for (size_t i = 0; i < total; i++) { float a = fabsf(residual[i]); if (a > mx) mx = a; }
    h->max_residual = mx;
    free(residual);
    return 0;
}

float ns_hierarchy_eval(const ns_hierarchy_t *h, float x, float y, float z)
{
    float r = 0.0f;
    for (uint8_t i = 0; i < h->num_levels; i++)
        if (h->valid[i]) r += ns_spline_3d_eval(&h->levels[i], x, y, z);
    return r;
}

sitton_f32_t ns_hierarchy_eval_sitton(const ns_hierarchy_t *h,
    sitton_f32_t x, sitton_f32_t y, sitton_f32_t z)
{
    sitton_f32_t r = sn_zero();
    for (uint8_t i = 0; i < h->num_levels; i++)
        if (h->valid[i]) r = sn_add(r, ns_spline_3d_eval_sitton(&h->levels[i], x, y, z));
    return r;
}

void ns_hierarchy_destroy(ns_hierarchy_t *h)
{
    for (uint8_t i = 0; i < NS_MAX_LEVELS; i++)
        if (h->valid[i]) ns_spline_3d_destroy(&h->levels[i]);
    memset(h, 0, sizeof(*h));
}

void ns_hierarchy_print_stats(const ns_hierarchy_t *h)
{
    float ratio = h->compressed_bytes ? (float)h->original_bytes / (float)h->compressed_bytes : 0;
    fprintf(stderr,
        " Neural Spline(tm) Hierarchy\n"
        "   Levels:     %u\n"
        "   Original:   %zu bytes\n"
        "   Compressed: %zu bytes\n"
        "   Ratio:      %.1f:1\n"
        "   RMS:        %e\n"
        "   Max:        %e\n",
        h->num_levels, h->original_bytes, h->compressed_bytes,
        ratio, h->rms_residual, h->max_residual);
}

void omega_field_eval(const omega_spline_field_t *field,
                      const omega_chip_geometry_t *geom,
                      float x, float y, float z,
                      float out_E[3], float out_H[3])
{
    (void)field; (void)geom; (void)x; (void)y; (void)z;
    out_E[0] = out_E[1] = out_E[2] = 0.0f;
    out_H[0] = out_H[1] = out_H[2] = 0.0f;
}

