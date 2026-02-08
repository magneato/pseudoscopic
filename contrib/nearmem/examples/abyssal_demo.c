/*
 * abyssal_demo.c - Demonstration of the Abyssal Circuit Debugger
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   "In the deepest trenches of the ocean, where crushing pressure
 *    and absolute darkness should make life impossible, creatures
 *    create their own light. They pulse, they glow, they communicate
 *    in a language of photons.
 *
 *    This debugger speaks the same language. Each signal pulse is
 *    a bioluminescent flash. Each clock edge is a heartbeat. Each
 *    fault is a predator lurking in the dark.
 *
 *    Welcome to the abyss."
 *
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This demo creates a test circuit and runs the Abyssal debugger on it.
 *
 * Circuit: 8-bit LFSR with tap monitoring and fault injection
 *
 * CONTROLS:
 *   Arrow keys / hjkl  - Navigate (time and signals)
 *   n / p              - Jump to next/previous edge
 *   + / -              - Zoom in/out
 *   Space              - Step simulation
 *   r                  - Run 100 cycles
 *   t                  - Trace selected signal forward
 *   1 / 2 / 3          - Switch views (Waveform/Trace/Faults)
 *   :                  - Enter command mode
 *   ?                  - Show help
 *   q                  - Quit
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "gpufpga.h"
#include "abyssal.h"

/*
 * Build test circuit: 8-bit LFSR with output register
 */
static int build_test_circuit(gpufpga_builder_t *b, gpufpga_ctx_t *ctx) {
    printf("Building test circuit (8-bit LFSR)...\n");
    
    /* Clock input */
    wire_id_t clk = gpufpga_builder_input(b, 0, 0);
    
    /* 8-bit LFSR shift register */
    wire_id_t q[8];
    
    /* Create flip-flops */
    for (int i = 0; i < 8; i++) {
        q[i] = gpufpga_builder_ff(b, 0, clk, WIRE_INVALID, WIRE_INVALID);
        
        /* Set initial value (seed = 0x01) */
        for (uint32_t j = 0; j < b->num_ffs; j++) {
            if (b->ffs[j].q_output == q[i]) {
                b->ffs[j].init_value = (i == 0) ? 1 : 0;
                break;
            }
        }
    }
    
    /* LFSR feedback: taps at 7, 5, 4, 3 (maximal length) */
    wire_id_t tap_75 = gpufpga_builder_xor2(b, q[7], q[5]);
    wire_id_t tap_43 = gpufpga_builder_xor2(b, q[4], q[3]);
    wire_id_t feedback = gpufpga_builder_xor2(b, tap_75, tap_43);
    
    /* Connect shift register */
    for (uint32_t j = 0; j < b->num_ffs; j++) {
        for (int i = 0; i < 8; i++) {
            if (b->ffs[j].q_output == q[i]) {
                b->ffs[j].d_input = (i == 0) ? feedback : q[i-1];
                break;
            }
        }
    }
    
    /* Create some combinational logic to monitor */
    wire_id_t q7_and_q0 = gpufpga_builder_and2(b, q[7], q[0]);
    wire_id_t any_high = gpufpga_builder_or2(b, q[0], q[1]);
    for (int i = 2; i < 8; i++) {
        any_high = gpufpga_builder_or2(b, any_high, q[i]);
    }
    
    /* Outputs */
    for (int i = 0; i < 8; i++) {
        gpufpga_builder_output(b, 1, i, q[i]);
    }
    gpufpga_builder_output(b, 2, 0, feedback);
    gpufpga_builder_output(b, 2, 1, q7_and_q0);
    gpufpga_builder_output(b, 2, 2, any_high);
    
    /* Build the circuit */
    int ret = gpufpga_builder_finish(b, ctx);
    if (ret == 0) {
        printf("Circuit built: %u LUTs, %u FFs, %u wires\n",
               ctx->header->num_luts, ctx->header->num_ffs, ctx->header->num_wires);
    }
    
    return ret;
}

/*
 * Print splash screen
 */
static void print_splash(void) {
    printf("\n");
    printf("\033[38;2;0;128;160m");  /* Dim cyan */
    printf("    ╔══════════════════════════════════════════════════════════════════════╗\n");
    printf("    ║                                                                      ║\n");
    printf("\033[38;2;0;255;255m");  /* Bright cyan */
    printf("    ║      ▄▄▄       ▄▄▄▄    ▓██   ██▓  ██████   ██████  ▄▄▄       ██▓     ║\n");
    printf("    ║     ▒████▄    ▓█████▄   ▒██  ██▒▒██    ▒ ▒██    ▒ ▒████▄    ▓██▒     ║\n");
    printf("    ║     ▒██  ▀█▄  ▒██▒ ▄██   ▒██ ██░░ ▓██▄   ░ ▓██▄   ▒██  ▀█▄  ▒██░     ║\n");
    printf("    ║     ░██▄▄▄▄██ ▒██░█▀     ░ ▐██▓░  ▒   ██▒  ▒   ██▒░██▄▄▄▄██ ▒██░     ║\n");
    printf("    ║      ▓█   ▓██▒░▓█  ▀█▓   ░ ██▒▓░▒██████▒▒▒██████▒▒ ▓█   ▓██▒░██████▒ ║\n");
    printf("    ║      ▒▒   ▓▒█░░▒▓███▀▒    ██▒▒▒ ▒ ▒▓▒ ▒ ░▒ ▒▓▒ ▒ ░ ▒▒   ▓▒█░░ ▒░▓  ░ ║\n");
    printf("\033[38;2;0;128;160m");  /* Dim cyan */
    printf("    ║                                                                      ║\n");
    printf("    ║           Deep-Sea Bioluminescent Circuit Debugger                   ║\n");
    printf("    ║                                                                      ║\n");
    printf("\033[38;2;100;120;140m");  /* Dim text */
    printf("    ║   \"In darkness, we find clarity. In silence, we hear signals.\"     ║\n");
    printf("    ║                                                                      ║\n");
    printf("\033[38;2;0;128;160m");
    printf("    ╚═════════════════════════════════════════════════════===══════════════╝\n");
    printf("\033[0m\n");
}

/*
 * Print controls
 */
static void print_controls(void) {
    printf("\033[38;2;0;200;200m");  /* Cyan */
    printf("  CONTROLS:\n");
    printf("\033[38;2;150;170;190m");  /* Light gray */
    printf("    ← → ↑ ↓  or  h j k l    Navigate (time and signals)\n");
    printf("    n / p                   Jump to next/previous signal edge\n");
    printf("    + / -                   Zoom in/out timeline\n");
    printf("    Space                   Step simulation (1 cycle)\n");
    printf("    r                       Run simulation (100 cycles)\n");
    printf("    t                       Trace signal forward through logic\n");
    printf("    1 / 2 / 3               Switch view (Waveform/Trace/Faults)\n");
    printf("    g / G                   Go to start/end\n");
    printf("    :                       Enter command mode\n");
    printf("    ?                       Show help in status bar\n");
    printf("    q                       Quit\n");
    printf("\033[0m\n");
}

int main(int argc, char *argv[]) {
    gpufpga_builder_t builder;
    gpufpga_ctx_t fpga_ctx;
    abyssal_ctx_t dbg_ctx;
    
    (void)argc; (void)argv;
    
    print_splash();
    print_controls();
    
    printf("\033[38;2;0;200;200m");
    printf("  Initializing...\n\n");
    printf("\033[0m");
    
    /* Initialize FPGA simulator */
    gpufpga_init(&fpga_ctx, NULL);
    gpufpga_builder_init(&builder, 500, 100);
    
    /* Build test circuit */
    if (build_test_circuit(&builder, &fpga_ctx) != 0) {
        fprintf(stderr, "Failed to build circuit\n");
        return 1;
    }
    gpufpga_builder_free(&builder);
    gpufpga_reset(&fpga_ctx);
    
    /* Initialize debugger */
    if (abyssal_init(&dbg_ctx, &fpga_ctx) != 0) {
        fprintf(stderr, "Failed to initialize debugger\n");
        gpufpga_shutdown(&fpga_ctx);
        return 1;
    }
    
    /* Add signals to watch */
    printf("  Adding signals to waveform view...\n");
    
    /* Find wires by their role (this is a bit hacky for demo) */
    /* In real usage, we'd have proper signal names from netlist */
    
    /* Add clock */
    abyssal_add_signal(&dbg_ctx, 0, "clk");
    
    /* Add LFSR bits (FF outputs are wires 1-8 approximately) */
    for (int i = 0; i < 8; i++) {
        char name[32];
        snprintf(name, sizeof(name), "lfsr[%d]", i);
        /* The FF outputs are the first wires allocated after inputs */
        abyssal_add_signal(&dbg_ctx, i + 1, name);
    }
    
    /* Add feedback signal */
    abyssal_add_signal(&dbg_ctx, 9, "feedback");
    
    /* Add some combinational signals */
    abyssal_add_signal(&dbg_ctx, fpga_ctx.header->num_wires - 3, "q7_and_q0");
    abyssal_add_signal(&dbg_ctx, fpga_ctx.header->num_wires - 2, "any_high");
    
    /* Run initial simulation to generate some waveform data */
    printf("  Running initial simulation (50 cycles)...\n");
    for (int i = 0; i < 50; i++) {
        gpufpga_step(&fpga_ctx);
        abyssal_capture_cycle(&dbg_ctx);
    }
    
    /* Analyze for faults */
    printf("  Analyzing circuit for faults...\n");
    int faults = abyssal_analyze_faults(&dbg_ctx);
    printf("  Found %d potential issues\n", faults);
    
    printf("\n\033[38;2;0;255;136m");  /* Green */
    printf("  Ready! Press Enter to launch debugger...\n");
    printf("\033[0m");
    getchar();
    
    /* Run debugger */
    abyssal_run(&dbg_ctx);
    
    /* Cleanup */
    abyssal_shutdown(&dbg_ctx);
    gpufpga_shutdown(&fpga_ctx);
    
    printf("\n");
    printf("\033[38;2;0;200;200m");
    printf("  Thank you for exploring the abyss.\n");
    printf("\033[38;2;100;120;140m");
    printf("  \"The light you saw was not the sun. It was something far more ancient.\"\n");
    printf("\033[0m\n");
    
    return 0;
}
