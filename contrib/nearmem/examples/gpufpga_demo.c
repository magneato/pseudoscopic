/*
 * gpufpga_demo.c - Demonstration of FPGA Emulation via Branchless GPU Computing
 *
 * "Hardware is just software that's hard to change."
 *
 * This demo shows:
 *   1. Building digital circuits programmatically (gates, flip-flops)
 *   2. Simulating them with branchless evaluation (GPU-friendly)
 *   3. All state in VRAM via near-memory (no copies)
 *
 * Circuits demonstrated:
 *   1. Basic gates (AND, OR, XOR, NOT)
 *   2. Full adder
 *   3. 8-bit counter
 *   4. 8-bit shift register (LFSR)
 *
 * The key insight: FPGA logic is INHERENTLY BRANCHLESS.
 * A LUT is just a table lookup. A flip-flop is a conditional move.
 * This maps perfectly to GPU execution.
 *
 * Copyright (C) 2025 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "nearmem.h"
#include "gpufpga.h"

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DEMO 1: Basic Gates
 * ════════════════════════════════════════════════════════════════════════════
 */

static void demo_basic_gates(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Demo 1: Basic Gates (AND, OR, XOR, NOT)                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    gpufpga_builder_t builder;
    gpufpga_ctx_t ctx;
    
    gpufpga_init(&ctx, NULL);
    gpufpga_builder_init(&builder, 100, 10);
    
    /* Create inputs */
    wire_id_t a = gpufpga_builder_input(&builder, 0, 0);  /* Port 0, bit 0 */
    wire_id_t b = gpufpga_builder_input(&builder, 0, 1);  /* Port 0, bit 1 */
    
    /* Create gates */
    wire_id_t and_out = gpufpga_builder_and2(&builder, a, b);
    wire_id_t or_out = gpufpga_builder_or2(&builder, a, b);
    wire_id_t xor_out = gpufpga_builder_xor2(&builder, a, b);
    wire_id_t not_a = gpufpga_builder_not(&builder, a);
    wire_id_t nand_out = gpufpga_builder_nand2(&builder, a, b);
    
    /* Create outputs */
    gpufpga_builder_output(&builder, 1, 0, and_out);   /* Port 1, bit 0: AND */
    gpufpga_builder_output(&builder, 1, 1, or_out);    /* Port 1, bit 1: OR */
    gpufpga_builder_output(&builder, 1, 2, xor_out);   /* Port 1, bit 2: XOR */
    gpufpga_builder_output(&builder, 1, 3, not_a);     /* Port 1, bit 3: NOT A */
    gpufpga_builder_output(&builder, 1, 4, nand_out);  /* Port 1, bit 4: NAND */
    
    /* Build circuit */
    gpufpga_builder_finish(&builder, &ctx);
    gpufpga_reset(&ctx);
    
    /* Test all input combinations */
    printf("A B | AND  OR XOR NOT_A NAND\n");
    printf("----+------------------------\n");
    
    for (int av = 0; av <= 1; av++) {
        for (int bv = 0; bv <= 1; bv++) {
            gpufpga_set_input(&ctx, 0, (bv << 1) | av);
            gpufpga_step(&ctx);
            
            uint64_t out = gpufpga_get_output(&ctx, 1);
            
            printf("%d %d |  %d    %d   %d    %d     %d\n",
                   av, bv,
                   (int)((out >> 0) & 1),  /* AND */
                   (int)((out >> 1) & 1),  /* OR */
                   (int)((out >> 2) & 1),  /* XOR */
                   (int)((out >> 3) & 1),  /* NOT A */
                   (int)((out >> 4) & 1)); /* NAND */
        }
    }
    
    printf("\nCircuit: %u LUTs, %u wires\n", 
           ctx.header->num_luts, ctx.header->num_wires);
    printf("Key point: Each LUT is a BRANCHLESS table lookup!\n");
    
    gpufpga_builder_free(&builder);
    gpufpga_shutdown(&ctx);
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DEMO 2: Full Adder
 * ════════════════════════════════════════════════════════════════════════════
 */

static void demo_full_adder(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Demo 2: Full Adder (A + B + Cin = {Cout, Sum})                  ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    gpufpga_builder_t builder;
    gpufpga_ctx_t ctx;
    
    gpufpga_init(&ctx, NULL);
    gpufpga_builder_init(&builder, 100, 10);
    
    /* Inputs */
    wire_id_t a = gpufpga_builder_input(&builder, 0, 0);
    wire_id_t b = gpufpga_builder_input(&builder, 0, 1);
    wire_id_t cin = gpufpga_builder_input(&builder, 0, 2);
    
    /* Full adder logic:
     * Sum = A XOR B XOR Cin
     * Cout = (A AND B) OR (Cin AND (A XOR B))
     */
    wire_id_t a_xor_b = gpufpga_builder_xor2(&builder, a, b);
    wire_id_t sum = gpufpga_builder_xor2(&builder, a_xor_b, cin);
    
    wire_id_t a_and_b = gpufpga_builder_and2(&builder, a, b);
    wire_id_t cin_and_axorb = gpufpga_builder_and2(&builder, cin, a_xor_b);
    wire_id_t cout = gpufpga_builder_or2(&builder, a_and_b, cin_and_axorb);
    
    /* Outputs */
    gpufpga_builder_output(&builder, 1, 0, sum);
    gpufpga_builder_output(&builder, 1, 1, cout);
    
    /* Build */
    gpufpga_builder_finish(&builder, &ctx);
    gpufpga_reset(&ctx);
    
    /* Test all combinations */
    printf("A B Cin | Sum Cout | Decimal\n");
    printf("--------+----------+---------\n");
    
    for (int av = 0; av <= 1; av++) {
        for (int bv = 0; bv <= 1; bv++) {
            for (int cv = 0; cv <= 1; cv++) {
                gpufpga_set_input(&ctx, 0, (cv << 2) | (bv << 1) | av);
                gpufpga_step(&ctx);
                
                uint64_t out = gpufpga_get_output(&ctx, 1);
                int s = (out >> 0) & 1;
                int co = (out >> 1) & 1;
                
                int expected = av + bv + cv;
                
                printf("%d %d  %d  |  %d   %d   |   %d %s\n",
                       av, bv, cv, s, co,
                       (co << 1) | s,
                       ((co << 1) | s) == expected ? "✓" : "✗");
            }
        }
    }
    
    printf("\nCircuit: %u LUTs, %u wires\n",
           ctx.header->num_luts, ctx.header->num_wires);
    
    gpufpga_builder_free(&builder);
    gpufpga_shutdown(&ctx);
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DEMO 3: 8-bit Counter
 * ════════════════════════════════════════════════════════════════════════════
 */

static void demo_counter(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Demo 3: 8-bit Counter (flip-flops + increment logic)            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    gpufpga_builder_t builder;
    gpufpga_ctx_t ctx;
    
    gpufpga_init(&ctx, NULL);
    gpufpga_builder_init(&builder, 500, 100);
    
    /* Clock and reset inputs */
    wire_id_t clock = gpufpga_builder_input(&builder, 0, 0);
    wire_id_t reset = gpufpga_builder_input(&builder, 0, 1);
    
    /* Build 8-bit ripple counter manually */
    wire_id_t q[8];
    wire_id_t toggle = gpufpga_builder_wire(&builder);  /* Constant 1 for toggle */
    
    /* Initialize first toggle signal */
    /* We need a constant 1 - use a LUT with all 1s */
    gpufpga_lut_t *const_lut = &builder.luts[builder.num_luts++];
    const_lut->lut_mask = 0xFFFFFFFFFFFFFFFFULL;  /* Always 1 */
    const_lut->inputs[0] = 0;  /* Don't care */
    const_lut->inputs[1] = 0;
    const_lut->inputs[2] = 0;
    const_lut->inputs[3] = 0;
    const_lut->inputs[4] = 0;
    const_lut->inputs[5] = 0;
    const_lut->output = toggle;
    const_lut->num_inputs = 1;
    const_lut->level = 0;
    
    /* Create 8 flip-flops with toggle logic */
    wire_id_t carry = toggle;  /* First bit always toggles */
    
    for (int i = 0; i < 8; i++) {
        /* D input = Q XOR enable (toggle when enabled) */
        /* For now, create placeholder and fix after */
        q[i] = gpufpga_builder_ff(&builder, 0, clock, WIRE_INVALID, reset);
    }
    
    /* Now create the toggle logic and connect */
    carry = toggle;
    for (int i = 0; i < 8; i++) {
        /* Toggle logic: d[i] = q[i] XOR carry
         * carry[i+1] = q[i] AND carry[i]
         */
        wire_id_t d_i = gpufpga_builder_xor2(&builder, q[i], carry);
        
        /* Update FF's d_input */
        for (uint32_t j = 0; j < builder.num_ffs; j++) {
            if (builder.ffs[j].q_output == q[i]) {
                builder.ffs[j].d_input = d_i;
                break;
            }
        }
        
        /* Update carry for next bit */
        if (i < 7) {
            carry = gpufpga_builder_and2(&builder, q[i], carry);
        }
    }
    
    /* Outputs */
    for (int i = 0; i < 8; i++) {
        gpufpga_builder_output(&builder, 1, i, q[i]);
    }
    
    /* Build */
    gpufpga_builder_finish(&builder, &ctx);
    gpufpga_reset(&ctx);
    
    printf("Simulating 20 clock cycles:\n\n");
    printf("Cycle | Count (binary)  | Decimal\n");
    printf("------+-----------------+---------\n");
    
    double start = get_time_ms();
    
    for (int cycle = 0; cycle < 20; cycle++) {
        /* Clock edge */
        gpufpga_set_input(&ctx, 0, 1);  /* clock=1, reset=0 */
        gpufpga_step(&ctx);
        
        uint64_t count = gpufpga_get_output(&ctx, 1);
        
        printf("  %2d  | ", cycle);
        for (int i = 7; i >= 0; i--) {
            printf("%d", (int)((count >> i) & 1));
        }
        printf(" | %3lu\n", count);
    }
    
    double elapsed = get_time_ms() - start;
    
    printf("\nCircuit: %u LUTs, %u FFs, %u wires\n",
           ctx.header->num_luts, ctx.header->num_ffs, ctx.header->num_wires);
    printf("Simulation: 20 cycles in %.2f ms\n", elapsed);
    
    /* Speed test */
    printf("\nSpeed test: 1,000,000 cycles...\n");
    gpufpga_reset(&ctx);
    
    start = get_time_ms();
    gpufpga_run(&ctx, 1000000);
    elapsed = get_time_ms() - start;
    
    printf("Time: %.2f ms\n", elapsed);
    printf("Rate: %.2f MHz equivalent\n", 1000000.0 / elapsed / 1000.0);
    printf("Final count: %lu (expected: %d)\n", 
           gpufpga_get_output(&ctx, 1), 1000000 % 256);
    
    gpufpga_print_stats(&ctx);
    
    gpufpga_builder_free(&builder);
    gpufpga_shutdown(&ctx);
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DEMO 4: Linear Feedback Shift Register (LFSR)
 * ════════════════════════════════════════════════════════════════════════════
 */

static void demo_lfsr(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Demo 4: 8-bit LFSR (pseudo-random number generator)             ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    gpufpga_builder_t builder;
    gpufpga_ctx_t ctx;
    
    gpufpga_init(&ctx, NULL);
    gpufpga_builder_init(&builder, 200, 50);
    
    /* Clock input */
    wire_id_t clock = gpufpga_builder_input(&builder, 0, 0);
    
    /* 8-bit shift register */
    wire_id_t q[8];
    
    /* Create flip-flops (will connect later) */
    for (int i = 0; i < 8; i++) {
        q[i] = gpufpga_builder_ff(&builder, 0, clock, WIRE_INVALID, WIRE_INVALID);
        
        /* Set initial value (seed) to 0x01 */
        for (uint32_t j = 0; j < builder.num_ffs; j++) {
            if (builder.ffs[j].q_output == q[i]) {
                builder.ffs[j].init_value = (i == 0) ? 1 : 0;
                break;
            }
        }
    }
    
    /* LFSR feedback: new_bit = q[7] XOR q[5] XOR q[4] XOR q[3]
     * (Taps for maximal length 8-bit LFSR) */
    wire_id_t tap1 = gpufpga_builder_xor2(&builder, q[7], q[5]);
    wire_id_t tap2 = gpufpga_builder_xor2(&builder, q[4], q[3]);
    wire_id_t feedback = gpufpga_builder_xor2(&builder, tap1, tap2);
    
    /* Connect shift register:
     * q[0] <- feedback
     * q[i] <- q[i-1] for i > 0
     */
    for (uint32_t j = 0; j < builder.num_ffs; j++) {
        for (int i = 0; i < 8; i++) {
            if (builder.ffs[j].q_output == q[i]) {
                builder.ffs[j].d_input = (i == 0) ? feedback : q[i-1];
                break;
            }
        }
    }
    
    /* Outputs */
    for (int i = 0; i < 8; i++) {
        gpufpga_builder_output(&builder, 1, i, q[i]);
    }
    
    /* Build */
    gpufpga_builder_finish(&builder, &ctx);
    gpufpga_reset(&ctx);
    
    printf("LFSR sequence (first 30 values):\n\n");
    
    uint8_t seen[256] = {0};
    int unique_count = 0;
    
    for (int cycle = 0; cycle < 30; cycle++) {
        gpufpga_step(&ctx);
        
        uint64_t val = gpufpga_get_output(&ctx, 1);
        
        if (!seen[val]) {
            seen[val] = 1;
            unique_count++;
        }
        
        printf("%3d: 0x%02lX (%3lu) ", cycle, val, val);
        
        /* Visual bar */
        printf("[");
        for (int i = 0; i < (int)(val / 8); i++) printf("#");
        for (int i = (int)(val / 8); i < 32; i++) printf(" ");
        printf("]\n");
    }
    
    /* Find period */
    printf("\nFinding LFSR period...\n");
    gpufpga_reset(&ctx);
    
    uint64_t initial;
    gpufpga_step(&ctx);
    initial = gpufpga_get_output(&ctx, 1);
    
    int period = 1;
    while (period < 300) {
        gpufpga_step(&ctx);
        if (gpufpga_get_output(&ctx, 1) == initial)
            break;
        period++;
    }
    
    printf("Period: %d (theoretical max for 8-bit LFSR: 255)\n", period);
    
    printf("\nCircuit: %u LUTs, %u FFs, %u wires\n",
           ctx.header->num_luts, ctx.header->num_ffs, ctx.header->num_wires);
    
    gpufpga_builder_free(&builder);
    gpufpga_shutdown(&ctx);
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DEMO 5: Branchless Evaluation Proof
 * ════════════════════════════════════════════════════════════════════════════
 */

static void demo_branchless_proof(void) {
    printf("\n╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  Demo 5: Why FPGA Simulation is BRANCHLESS                       ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("The Key Insight:\n");
    printf("────────────────\n\n");
    
    printf("A Look-Up Table (LUT) implements ANY Boolean function:\n\n");
    
    printf("  Traditional (BRANCHING) implementation of AND:\n");
    printf("    if (a && b) return 1; else return 0;\n\n");
    
    printf("  LUT (BRANCHLESS) implementation:\n");
    printf("    uint8_t and_table[4] = {0, 0, 0, 1}; // Truth table\n");
    printf("    return and_table[(b << 1) | a];      // Just array index!\n\n");
    
    printf("This works for ANY function:\n\n");
    
    printf("  Function   | Truth Table | LUT Mask\n");
    printf("  -----------+-------------+---------\n");
    printf("  AND(a,b)   | 0001        | 0x8\n");
    printf("  OR(a,b)    | 0111        | 0xE\n");
    printf("  XOR(a,b)   | 0110        | 0x6\n");
    printf("  NAND(a,b)  | 1110        | 0x7\n");
    printf("  MUX(a,b,s) | 11001010    | 0xCA\n\n");
    
    printf("The GPU evaluation is:\n\n");
    printf("  __device__ uint8_t eval_lut(uint64_t mask, uint8_t inputs) {\n");
    printf("      return (mask >> inputs) & 1;  // NO BRANCHES!\n");
    printf("  }\n\n");
    
    printf("Similarly, flip-flop update is branchless:\n\n");
    printf("  // Traditional (branching):\n");
    printf("  if (clock_edge) q = d;\n\n");
    printf("  // Branchless:\n");
    printf("  q_next = (d & clock_edge) | (q & ~clock_edge);\n\n");
    
    printf("This means:\n");
    printf("  • Every LUT in the FPGA can be evaluated IN PARALLEL\n");
    printf("  • No warp divergence on GPU (all threads do same operation)\n");
    printf("  • Memory access patterns are predictable (coalesced)\n");
    printf("  • Perfect for GPU architecture!\n\n");
    
    printf("GPU Mapping:\n");
    printf("────────────\n");
    printf("  • 1 CUDA thread → 1 LUT (or group of LUTs)\n");
    printf("  • Shared memory → Wire state\n");
    printf("  • __syncthreads() → Clock edge\n");
    printf("  • Global memory → Waveform capture\n\n");
    
    printf("This is why gpuFPGA can be FAST:\n");
    printf("  • 10,000+ parallel LUT evaluations\n");
    printf("  • 700+ GB/s wire state bandwidth (HBM2)\n");
    printf("  • No branch prediction misses\n");
    printf("  • Deterministic execution time\n");
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MAIN
 * ════════════════════════════════════════════════════════════════════════════
 */

int main(int argc, char *argv[]) {
    (void)argc; (void)argv;
    
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  gpuFPGA: FPGA Emulation via Branchless GPU Computing            ║\n");
    printf("║                                                                  ║\n");
    printf("║  \"Hardware is just software that's hard to change.\"              ║\n");
    printf("║                                        - Someone wise            ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    
    demo_basic_gates();
    demo_full_adder();
    demo_counter();
    demo_lfsr();
    demo_branchless_proof();
    
    printf("\n");
    printf("════════════════════════════════════════════════════════════════════\n");
    printf("                         SUMMARY\n");
    printf("════════════════════════════════════════════════════════════════════\n\n");
    
    printf("What we demonstrated:\n\n");
    
    printf("1. CIRCUIT BUILDING\n");
    printf("   • Programmatic netlist construction\n");
    printf("   • LUTs for combinational logic\n");
    printf("   • Flip-flops for sequential logic\n\n");
    
    printf("2. BRANCHLESS EVALUATION\n");
    printf("   • LUT = table lookup (no if-else)\n");
    printf("   • FF = conditional move (no branches)\n");
    printf("   • Perfect for GPU execution\n\n");
    
    printf("3. REAL CIRCUITS\n");
    printf("   • Basic gates (AND, OR, XOR, NOT, NAND)\n");
    printf("   • Full adder (combinational)\n");
    printf("   • 8-bit counter (sequential)\n");
    printf("   • 8-bit LFSR (pseudo-random)\n\n");
    
    printf("THE IMPLICATIONS:\n\n");
    
    printf("• ANY digital circuit can be simulated this way\n");
    printf("• CPU designs (RISC-V, ARM) can be simulated\n");
    printf("• Neural network accelerators can be emulated\n");
    printf("• The GPU becomes a universal logic simulator\n\n");
    
    printf("THE STACK:\n\n");
    
    printf("  ┌─────────────────────────────────────────────────┐\n");
    printf("  │  Verilog/VHDL Design                            │\n");
    printf("  │       ↓                                         │\n");
    printf("  │  Technology Mapping (→ LUT/FF netlist)          │\n");
    printf("  │       ↓                                         │\n");
    printf("  │  gpuFPGA Simulation (branchless GPU)            │\n");
    printf("  │       ↓                                         │\n");
    printf("  │  Near-Memory (state in VRAM)                    │\n");
    printf("  │       ↓                                         │\n");
    printf("  │  Pseudoscopic (VRAM as block device)            │\n");
    printf("  │       ↓                                         │\n");
    printf("  │  Physical GPU (just transistors)                │\n");
    printf("  └─────────────────────────────────────────────────┘\n\n");
    
    printf("Cookie Monster's Unified Theory of Computing:\n");
    printf("  \"There is no GPU. There is no CPU. There is no FPGA.\n");
    printf("   There is only math, memory, and cookies.\" 🍪\n\n");
    
    return 0;
}
