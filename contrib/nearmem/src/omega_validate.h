#ifndef OMEGA_VALIDATE_H
#define OMEGA_VALIDATE_H
/*
 * omega_validate.h — gpuFPGA Circuit Validation
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_vram.h"
#include "omega_qxor.h"

#ifdef __cplusplus
extern "C" {
#endif

void omega_validate_circuit(omega_vram_layout_t *vram,
                            const omega_coupling_graph_t *graph,
                            const char *control_verilog);

#ifdef __cplusplus
}
#endif

#endif /* OMEGA_VALIDATE_H */

