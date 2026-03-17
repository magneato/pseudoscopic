/*
 * omega_entropy.c — Hardware Entropy via TruRNG3
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_entropy.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#include <sys/stat.h>

static const char *trng_paths[] = {
    "/dev/TrueRNG0", "/dev/TrueRNG", "/dev/ttyACM0", "/dev/ttyACM1", NULL
};

static int trng_refill(omega_entropy_t *ent)
{
    ssize_t nr = read(ent->fd, ent->buffer, TRNG_BUF_SIZE);
    if (nr <= 0) { nr = read(ent->fd, ent->buffer, 64); if (nr <= 0) return -1; }
    ent->buf_len = (size_t)nr;
    ent->buf_pos = 0;
    ent->refill_count++;
    for (size_t i = 0; i < ent->buf_len; i++) ent->byte_histogram[ent->buffer[i]]++;
    return 0;
}

int omega_entropy_init(omega_entropy_t *ent)
{
    return omega_entropy_init_device(ent, NULL, "omega_seeds.bin");
}

int omega_entropy_init_device(omega_entropy_t *ent, const char *dev, const char *log)
{
    memset(ent, 0, sizeof(*ent));
    ent->fd = -1; ent->seed_log_fd = -1; ent->source = TRNG_SRC_CLOSED;

    if (dev) {
        ent->fd = open(dev, O_RDONLY);
        if (ent->fd >= 0) ent->source = TRNG_SRC_HARDWARE;
    } else {
        for (const char **p = trng_paths; *p; p++) {
            ent->fd = open(*p, O_RDONLY);
            if (ent->fd >= 0) { ent->source = TRNG_SRC_HARDWARE; break; }
        }
    }
    if (ent->fd < 0) {
        ent->fd = open("/dev/urandom", O_RDONLY);
        if (ent->fd < 0) return -1;
        ent->source = TRNG_SRC_URANDOM;
        fprintf(stderr,
            "  [omega] TruRNG3 not found — using /dev/urandom\n"
            "  Results explore pseudorandom manifold only.\n"
            "  Connect TruRNG3 for genuine probability-space exploration.\n");
    }
    if (log) {
        ent->seed_log_fd = open(log, O_WRONLY | O_CREAT | O_TRUNC, S_IRUSR | S_IWUSR);
        if (ent->seed_log_fd < 0)
            fprintf(stderr, "  [omega] WARNING: cannot open seed log %s: %s\n", log, strerror(errno));
    }
    trng_refill(ent);
    return 0;
}

void omega_entropy_shutdown(omega_entropy_t *ent)
{
    if (ent->seed_log_fd >= 0) { fsync(ent->seed_log_fd); close(ent->seed_log_fd); }
    if (ent->fd >= 0) close(ent->fd);
    explicit_bzero(ent->buffer, TRNG_BUF_SIZE);
    ent->source = TRNG_SRC_CLOSED; ent->fd = -1; ent->seed_log_fd = -1;
}

size_t omega_entropy_read(omega_entropy_t *ent, void *out, size_t len)
{
    uint8_t *dst = (uint8_t *)out;
    size_t rem = len;
    while (rem > 0) {
        if (ent->buf_pos >= ent->buf_len) {
            if (trng_refill(ent) != 0) {
                if (ent->source == TRNG_SRC_HARDWARE) ent->source = TRNG_SRC_DEGRADED;
                struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
                uint64_t mix = (uint64_t)ts.tv_nsec ^ ((uint64_t)ts.tv_sec << 32);
                for (size_t i = 0; i < rem && i < 8; i++) dst[i] = ((uint8_t *)&mix)[i];
                ent->bytes_fallback += rem; ent->bytes_consumed += rem;
                return len;
            }
        }
        size_t avail = ent->buf_len - ent->buf_pos;
        size_t chunk = (rem < avail) ? rem : avail;
        memcpy(dst, ent->buffer + ent->buf_pos, chunk);
        if (ent->seed_log_fd >= 0) (void)write(ent->seed_log_fd, ent->buffer + ent->buf_pos, chunk);
        dst += chunk; ent->buf_pos += chunk; rem -= chunk;
        if (ent->source == TRNG_SRC_HARDWARE) ent->bytes_hw += chunk;
        else ent->bytes_fallback += chunk;
    }
    ent->bytes_consumed += len;
    return len;
}

uint64_t omega_entropy_u64(omega_entropy_t *ent)
{ uint64_t v; omega_entropy_read(ent, &v, sizeof(v)); return v; }

float omega_entropy_uniform(omega_entropy_t *ent)
{ uint32_t r; omega_entropy_read(ent, &r, sizeof(r)); return (float)(r >> 8) * 0x1.0p-24f; }

float omega_entropy_range(omega_entropy_t *ent, float lo, float hi)
{ return lo + omega_entropy_uniform(ent) * (hi - lo); }

void omega_entropy_normal_pair(omega_entropy_t *ent, float *n1, float *n2)
{
    float u1; do { u1 = omega_entropy_uniform(ent); } while (u1 < 1e-30f);
    float u2 = omega_entropy_uniform(ent);
    float r = sqrtf(-2.0f * logf(u1)), th = 2.0f * (float)M_PI * u2;
    *n1 = r * cosf(th); *n2 = r * sinf(th);
}

sitton_f32_t omega_entropy_sitton(omega_entropy_t *ent, float nominal, float scale)
{
    float u = omega_entropy_uniform(ent) * 2.0f - 1.0f;
    return (sitton_f32_t){ .real = nominal, .prob = scale * u };
}

sitton_f32_t omega_entropy_sitton_normal(omega_entropy_t *ent, float nominal, float sigma)
{
    float n1, n2; omega_entropy_normal_pair(ent, &n1, &n2); (void)n2;
    return (sitton_f32_t){ .real = nominal, .prob = sigma * n1 };
}

float omega_entropy_chi2(const omega_entropy_t *ent)
{
    if (ent->bytes_consumed < 256) return 0.0f;
    double expected = (double)ent->bytes_consumed / 256.0, chi2 = 0.0;
    for (int i = 0; i < 256; i++) {
        double d = (double)ent->byte_histogram[i] - expected;
        chi2 += (d * d) / expected;
    }
    return (float)chi2;
}

const char *omega_entropy_source_name(trng_source_t src)
{
    switch (src) {
    case TRNG_SRC_HARDWARE: return "TruRNG3 (hardware)";
    case TRNG_SRC_URANDOM:  return "/dev/urandom (kernel CSPRNG)";
    case TRNG_SRC_DEGRADED: return "DEGRADED (timestamp mixing)";
    case TRNG_SRC_CLOSED:   return "CLOSED";
    }
    return "UNKNOWN";
}

void omega_entropy_print_stats(const omega_entropy_t *ent)
{
    fprintf(stderr,
        "  [trng] Source: %s | Consumed: %lu | HW: %lu | Fallback: %lu | chi2: %.1f\n",
        omega_entropy_source_name(ent->source),
        (unsigned long)ent->bytes_consumed, (unsigned long)ent->bytes_hw,
        (unsigned long)ent->bytes_fallback, ent->last_chi2);
}

