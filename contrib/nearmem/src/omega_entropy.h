#ifndef OMEGA_ENTROPY_H
#define OMEGA_ENTROPY_H
/*
 * omega_entropy.h — Hardware Entropy via TruRNG3
 *
 * PRNG explores one deterministic sub-manifold.
 * Hardware entropy accesses the actual probability space.
 * The difference is not academic.
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include <stdint.h>
#include <stddef.h>
#include "omega_sitton.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    TRNG_SRC_HARDWARE,      /* TruRNG3 connected and streaming   */
    TRNG_SRC_URANDOM,       /* Fallback to /dev/urandom          */
    TRNG_SRC_DEGRADED,      /* Pool exhausted, timestamp mixing  */
    TRNG_SRC_CLOSED
} trng_source_t;

#define TRNG_BUF_SIZE 4096

typedef struct {
    int             fd;
    int             seed_log_fd;
    trng_source_t   source;
    uint8_t         buffer[TRNG_BUF_SIZE] __attribute__((aligned(64)));
    size_t          buf_pos;
    size_t          buf_len;
    uint64_t        bytes_consumed;
    uint64_t        bytes_hw;
    uint64_t        bytes_fallback;
    uint64_t        refill_count;
    uint64_t        byte_histogram[256];
    float           last_chi2;
    bool            bias_warned;
} omega_entropy_t;

int         omega_entropy_init(omega_entropy_t *ent);
int         omega_entropy_init_device(omega_entropy_t *ent, const char *dev, const char *log);
void        omega_entropy_shutdown(omega_entropy_t *ent);
size_t      omega_entropy_read(omega_entropy_t *ent, void *out, size_t len);
uint64_t    omega_entropy_u64(omega_entropy_t *ent);
float       omega_entropy_uniform(omega_entropy_t *ent);
float       omega_entropy_range(omega_entropy_t *ent, float lo, float hi);
void        omega_entropy_normal_pair(omega_entropy_t *ent, float *n1, float *n2);
sitton_f32_t omega_entropy_sitton(omega_entropy_t *ent, float nominal, float scale);
sitton_f32_t omega_entropy_sitton_normal(omega_entropy_t *ent, float nominal, float sigma);
float       omega_entropy_chi2(const omega_entropy_t *ent);
const char *omega_entropy_source_name(trng_source_t src);
void        omega_entropy_print_stats(const omega_entropy_t *ent);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_ENTROPY_H */

