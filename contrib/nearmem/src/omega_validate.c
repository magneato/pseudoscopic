/*
 * omega_validate.c — gpuFPGA Circuit Validation
 *
 * (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
 * Patent Pending — All Rights Reserved
 */

#include "omega_validate.h"
#include "gpufpga.h"
#include <stdio.h>

static uint64_t coupling_to_pulse_amplitude(float coupling_g)
{
    /* Map coupling strength (GHz) to pulse amplitude.
     * Scale: 1 GHz coupling → 0x8000 pulse units (14-bit DAC). */
    float clamped = (coupling_g > 1.0f) ? 1.0f : (coupling_g < -1.0f) ? -1.0f : coupling_g;
    return (uint64_t)((clamped + 1.0f) * 0.5f * 0x3FFF);
}

void omega_validate_circuit(omega_vram_layout_t *vram,
                            const omega_coupling_graph_t *graph,
                            const char *control_verilog)
{
    gpufpga_ctx_t fpga;
    gpufpga_hdl_options_t opts = GPUFPGA_HDL_OPTIONS_DEFAULT;
    gpufpga_hdl_result_t result;

    gpufpga_init(&fpga, &vram->nm_ctx);
    gpufpga_load_verilog(&fpga, control_verilog, &opts, &result);

    /* Inject coupling strengths as pulse amplitudes */
    for (size_t i = 0; i < graph->num_couplings; i++) {
        uint64_t amp = coupling_to_pulse_amplitude(graph->couplings[i].coupling_g);
        gpufpga_set_input(&fpga, graph->couplings[i].qubit_a, amp);
    }

    /* Run 10,000 clock cycles per gate */
    gpufpga_run(&fpga, 10000);

    /* Extract fidelity metrics */
    for (size_t q = 0; q < graph->num_qubits; q++) {
        uint64_t readout = gpufpga_get_output(&fpga, (uint32_t)q);
        fprintf(stderr, "  [validate] Qubit %zu fidelity proxy: 0x%016lx\n", q, readout);
    }

    gpufpga_save_vcd(&fpga, "omega_circuit_validation.vcd");
    gpufpga_shutdown(&fpga);
}

