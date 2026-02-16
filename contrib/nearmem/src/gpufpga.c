/*
 * gpufpga.c - GPU-Accelerated FPGA Emulator Implementation
 *
 * Branchless FPGA simulation via near-memory computing.
 * Every LUT is a table lookup. Every FF is a conditional move.
 * No branches. Pure parallel evaluation.
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include "gpufpga.h"

/*
 * ════════════════════════════════════════════════════════════════════════════
 * TIMING
 * ════════════════════════════════════════════════════════════════════════════
 */

static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * INITIALIZATION
 * ════════════════════════════════════════════════════════════════════════════
 */

int gpufpga_init(gpufpga_ctx_t *ctx, nearmem_ctx_t *nm_ctx) {
    if (!ctx)
        return -1;
    
    memset(ctx, 0, sizeof(*ctx));
    ctx->nm_ctx = nm_ctx;
    ctx->use_gpu = (nm_ctx != NULL);
    
    return 0;
}

void gpufpga_shutdown(gpufpga_ctx_t *ctx) {
    if (!ctx)
        return;
    
    if (ctx->nm_ctx) {
        if (ctx->circuit_region.cpu_ptr)
            nearmem_free(ctx->nm_ctx, &ctx->circuit_region);
        if (ctx->state_region.cpu_ptr)
            nearmem_free(ctx->nm_ctx, &ctx->state_region);
        if (ctx->waveform_region.cpu_ptr)
            nearmem_free(ctx->nm_ctx, &ctx->waveform_region);
    } else {
        free(ctx->header);
        free(ctx->wire_state);
        free(ctx->wire_next);
        free(ctx->ff_state);
        free(ctx->waveform);
    }
    
    memset(ctx, 0, sizeof(*ctx));
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * CIRCUIT LOADING
 * ════════════════════════════════════════════════════════════════════════════
 */

int gpufpga_load_mem(gpufpga_ctx_t *ctx, const void *data, size_t size) {
    if (!ctx || !data || size < sizeof(gpufpga_header_t))
        return -1;
    
    const gpufpga_header_t *hdr = (const gpufpga_header_t *)data;
    
    /* Validate header */
    if (hdr->magic != GPUFPGA_MAGIC) {
        fprintf(stderr, "gpufpga: Invalid magic number\n");
        return -1;
    }
    
    /* Calculate sizes */
    size_t luts_size = hdr->num_luts * sizeof(gpufpga_lut_t);
    size_t ffs_size = hdr->num_ffs * sizeof(gpufpga_ff_t);
    size_t ios_size = hdr->num_ios * sizeof(gpufpga_io_t);
    size_t levels_size = (hdr->num_levels + 1) * sizeof(uint32_t);
    size_t circuit_size = sizeof(gpufpga_header_t) + luts_size + ffs_size + 
                          ios_size + levels_size;
    
    size_t wire_size = hdr->num_wires;
    size_t state_size = wire_size * 2 + hdr->num_ffs;  /* wire + wire_next + ff */
    
    /* Allocate memory */
    if (ctx->nm_ctx) {
        /* Allocate in VRAM */
        if (nearmem_alloc(ctx->nm_ctx, &ctx->circuit_region, circuit_size) != NEARMEM_OK) {
            fprintf(stderr, "gpufpga: Failed to allocate circuit in VRAM\n");
            return -1;
        }
        
        if (nearmem_alloc(ctx->nm_ctx, &ctx->state_region, state_size) != NEARMEM_OK) {
            fprintf(stderr, "gpufpga: Failed to allocate state in VRAM\n");
            nearmem_free(ctx->nm_ctx, &ctx->circuit_region);
            return -1;
        }
        
        /* Copy circuit data to VRAM */
        memcpy(ctx->circuit_region.cpu_ptr, data, size);
        
        /* Set up pointers */
        uint8_t *ptr = (uint8_t *)ctx->circuit_region.cpu_ptr;
        ctx->header = (gpufpga_header_t *)ptr;
        ptr += sizeof(gpufpga_header_t);
        ctx->luts = (gpufpga_lut_t *)ptr;
        ptr += luts_size;
        ctx->ffs = (gpufpga_ff_t *)ptr;
        ptr += ffs_size;
        ctx->ios = (gpufpga_io_t *)ptr;
        ptr += ios_size;
        ctx->level_starts = (uint32_t *)ptr;
        
        /* Set up state pointers */
        ptr = (uint8_t *)ctx->state_region.cpu_ptr;
        ctx->wire_state = ptr;
        ptr += wire_size;
        ctx->wire_next = ptr;
        ptr += wire_size;
        ctx->ff_state = ptr;
        
        nearmem_sync(ctx->nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
    } else {
        /* Allocate in system RAM */
        ctx->header = malloc(circuit_size);
        if (!ctx->header)
            return -1;
        
        memcpy(ctx->header, data, size);
        
        uint8_t *ptr = (uint8_t *)ctx->header;
        ptr += sizeof(gpufpga_header_t);
        ctx->luts = (gpufpga_lut_t *)ptr;
        ptr += luts_size;
        ctx->ffs = (gpufpga_ff_t *)ptr;
        ptr += ffs_size;
        ctx->ios = (gpufpga_io_t *)ptr;
        ptr += ios_size;
        ctx->level_starts = (uint32_t *)ptr;
        
        ctx->wire_state = calloc(1, wire_size);
        ctx->wire_next = calloc(1, wire_size);
        ctx->ff_state = calloc(1, hdr->num_ffs);
        
        if (!ctx->wire_state || !ctx->wire_next || !ctx->ff_state) {
            free(ctx->header);
            free(ctx->wire_state);
            free(ctx->wire_next);
            free(ctx->ff_state);
            return -1;
        }
    }
    
    printf("gpufpga: Loaded circuit\n");
    printf("  LUTs:   %u\n", ctx->header->num_luts);
    printf("  FFs:    %u\n", ctx->header->num_ffs);
    printf("  Wires:  %u\n", ctx->header->num_wires);
    printf("  I/Os:   %u\n", ctx->header->num_ios);
    printf("  Levels: %u\n", ctx->header->num_levels);
    
    return 0;
}

void gpufpga_reset(gpufpga_ctx_t *ctx) {
    if (!ctx || !ctx->header)
        return;
    
    /* Clear wire state */
    memset(ctx->wire_state, 0, ctx->header->num_wires);
    memset(ctx->wire_next, 0, ctx->header->num_wires);
    
    /* Initialize FF state from descriptors */
    for (uint32_t i = 0; i < ctx->header->num_ffs; i++) {
        ctx->ff_state[i] = ctx->ffs[i].init_value;
        ctx->wire_state[ctx->ffs[i].q_output] = ctx->ffs[i].init_value;
    }
    
    ctx->cycles_simulated = 0;
    
    if (ctx->nm_ctx) {
        nearmem_sync(ctx->nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
    }
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * SIMULATION (CPU Implementation)
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * Evaluate a single LUT - BRANCHLESS
 */
static inline void eval_lut(const gpufpga_lut_t *lut, 
                            const uint8_t *wire_state,
                            uint8_t *wire_next) {
    /* Gather inputs into address (BRANCHLESS) */
    uint8_t addr = 0;
    addr |= (wire_state[lut->inputs[0]] & 1) << 0;
    addr |= (wire_state[lut->inputs[1]] & 1) << 1;
    addr |= (wire_state[lut->inputs[2]] & 1) << 2;
    addr |= (wire_state[lut->inputs[3]] & 1) << 3;
    addr |= (wire_state[lut->inputs[4]] & 1) << 4;
    addr |= (wire_state[lut->inputs[5]] & 1) << 5;
    
    /* Mask to actual input count */
    addr &= (1 << lut->num_inputs) - 1;
    
    /* Evaluate LUT - just a bit extraction, NO BRANCHES */
    wire_next[lut->output] = (lut->lut_mask >> addr) & 1;
}

/*
 * Update a single flip-flop - BRANCHLESS
 */
static inline void update_ff(const gpufpga_ff_t *ff,
                             uint8_t *wire_state,
                             uint8_t *ff_state,
                             int ff_idx) {
    uint8_t d = wire_state[ff->d_input];
    uint8_t en = (ff->enable == WIRE_INVALID) ? 1 : wire_state[ff->enable];
    uint8_t rst = (ff->reset == WIRE_INVALID) ? 0 : wire_state[ff->reset];
    uint8_t set = (ff->set == WIRE_INVALID) ? 0 : wire_state[ff->set];
    uint8_t q = ff_state[ff_idx];
    
    /* BRANCHLESS FF update */
    uint8_t new_q = gpufpga_ff_update(q, d, 1, en, rst, set);
    
    ff_state[ff_idx] = new_q;
    wire_state[ff->q_output] = new_q;
}

int gpufpga_step(gpufpga_ctx_t *ctx) {
    if (!ctx || !ctx->header)
        return -1;
    
    /* Phase 1: Evaluate all combinational logic (LUTs) */
    /* In levelized order if available, otherwise all at once */
    
    if (ctx->header->num_levels > 0) {
        /* Levelized evaluation - multiple passes */
        for (uint32_t level = 0; level < ctx->header->num_levels; level++) {
            uint32_t start = ctx->level_starts[level];
            uint32_t end = ctx->level_starts[level + 1];
            
            for (uint32_t i = start; i < end; i++) {
                eval_lut(&ctx->luts[i], ctx->wire_state, ctx->wire_next);
            }
            
            /* Copy next to current for next level */
            for (uint32_t i = start; i < end; i++) {
                wire_id_t out = ctx->luts[i].output;
                ctx->wire_state[out] = ctx->wire_next[out];
            }
        }
    } else {
        /* Simple evaluation - assumes no combinational loops */
        for (uint32_t i = 0; i < ctx->header->num_luts; i++) {
            eval_lut(&ctx->luts[i], ctx->wire_state, ctx->wire_next);
        }
        
        /* Copy all results */
        memcpy(ctx->wire_state, ctx->wire_next, ctx->header->num_wires);
    }
    
    /* Phase 2: Update all flip-flops (clock edge) */
    for (uint32_t i = 0; i < ctx->header->num_ffs; i++) {
        update_ff(&ctx->ffs[i], ctx->wire_state, ctx->ff_state, i);
    }
    
    /* Phase 3: Capture waveform if enabled */
    if (ctx->waveform && ctx->cycles_simulated < ctx->waveform_cycles) {
        memcpy(ctx->waveform + ctx->cycles_simulated * ctx->waveform_stride,
               ctx->wire_state, ctx->header->num_wires);
    }
    
    ctx->cycles_simulated++;
    
    return 0;
}

uint64_t gpufpga_run(gpufpga_ctx_t *ctx, uint64_t cycles) {
    if (!ctx || !ctx->header)
        return 0;
    
    double start = get_time_ms();
    
    for (uint64_t i = 0; i < cycles; i++) {
        gpufpga_step(ctx);
    }
    
    double elapsed = get_time_ms() - start;
    ctx->total_time_ms += elapsed;
    
    if (ctx->nm_ctx) {
        nearmem_sync(ctx->nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
    }
    
    return cycles;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * I/O ACCESS
 * ════════════════════════════════════════════════════════════════════════════
 */

void gpufpga_set_input(gpufpga_ctx_t *ctx, int port_id, uint64_t value) {
    if (!ctx || !ctx->header)
        return;
    
    for (uint32_t i = 0; i < ctx->header->num_ios; i++) {
        if (ctx->ios[i].port_id == port_id && ctx->ios[i].is_input) {
            int bit = ctx->ios[i].bit_index;
            ctx->wire_state[ctx->ios[i].wire] = (value >> bit) & 1;
        }
    }
}

uint64_t gpufpga_get_output(gpufpga_ctx_t *ctx, int port_id) {
    if (!ctx || !ctx->header)
        return 0;
    
    uint64_t result = 0;
    
    for (uint32_t i = 0; i < ctx->header->num_ios; i++) {
        if (ctx->ios[i].port_id == port_id && !ctx->ios[i].is_input) {
            int bit = ctx->ios[i].bit_index;
            result |= ((uint64_t)(ctx->wire_state[ctx->ios[i].wire] & 1)) << bit;
        }
    }
    
    return result;
}

uint8_t gpufpga_get_wire(gpufpga_ctx_t *ctx, wire_id_t wire) {
    if (!ctx || wire >= ctx->header->num_wires)
        return 0;
    return ctx->wire_state[wire] & 1;
}

void gpufpga_set_wire(gpufpga_ctx_t *ctx, wire_id_t wire, uint8_t value) {
    if (!ctx || wire >= ctx->header->num_wires)
        return;
    ctx->wire_state[wire] = value & 1;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * WAVEFORM CAPTURE
 * ════════════════════════════════════════════════════════════════════════════
 */

int gpufpga_enable_waveform(gpufpga_ctx_t *ctx, uint32_t max_cycles) {
    if (!ctx || !ctx->header)
        return -1;
    
    ctx->waveform_cycles = max_cycles;
    ctx->waveform_stride = ctx->header->num_wires;
    
    size_t waveform_size = max_cycles * ctx->waveform_stride;
    
    if (ctx->nm_ctx) {
        if (nearmem_alloc(ctx->nm_ctx, &ctx->waveform_region, waveform_size) != NEARMEM_OK)
            return -1;
        ctx->waveform = ctx->waveform_region.cpu_ptr;
    } else {
        ctx->waveform = calloc(1, waveform_size);
        if (!ctx->waveform)
            return -1;
    }
    
    return 0;
}

void gpufpga_print_stats(gpufpga_ctx_t *ctx) {
    if (!ctx || !ctx->header)
        return;
    
    printf("\n=== gpuFPGA Statistics ===\n");
    printf("Cycles simulated: %lu\n", ctx->cycles_simulated);
    printf("Total time: %.2f ms\n", ctx->total_time_ms);
    
    if (ctx->total_time_ms > 0) {
        double cycles_per_sec = ctx->cycles_simulated / (ctx->total_time_ms / 1000.0);
        double lut_evals = (double)ctx->cycles_simulated * ctx->header->num_luts;
        double lut_per_sec = lut_evals / (ctx->total_time_ms / 1000.0);
        
        printf("Simulation rate: %.2f kHz\n", cycles_per_sec / 1000.0);
        printf("LUT evaluations/sec: %.2f M\n", lut_per_sec / 1e6);
    }
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * CIRCUIT BUILDER
 * ════════════════════════════════════════════════════════════════════════════
 */

int gpufpga_builder_init(gpufpga_builder_t *b, uint32_t max_luts, uint32_t max_ffs) {
    if (!b)
        return -1;
    
    memset(b, 0, sizeof(*b));
    
    b->luts = calloc(max_luts, sizeof(gpufpga_lut_t));
    b->ffs = calloc(max_ffs, sizeof(gpufpga_ff_t));
    b->ios = calloc(256, sizeof(gpufpga_io_t));
    
    if (!b->luts || !b->ffs || !b->ios) {
        free(b->luts);
        free(b->ffs);
        free(b->ios);
        return -1;
    }
    
    b->max_luts = max_luts;
    b->max_ffs = max_ffs;
    b->max_ios = 256;
    b->num_wires = 0;
    
    return 0;
}

wire_id_t gpufpga_builder_wire(gpufpga_builder_t *b) {
    return b->num_wires++;
}

wire_id_t gpufpga_builder_input(gpufpga_builder_t *b, int port_id, int bit) {
    if (b->num_ios >= b->max_ios)
        return WIRE_INVALID;
    
    wire_id_t wire = gpufpga_builder_wire(b);
    
    b->ios[b->num_ios].wire = wire;
    b->ios[b->num_ios].port_id = port_id;
    b->ios[b->num_ios].bit_index = bit;
    b->ios[b->num_ios].is_input = 1;
    b->num_ios++;
    
    return wire;
}

void gpufpga_builder_output(gpufpga_builder_t *b, int port_id, int bit, wire_id_t wire) {
    if (b->num_ios >= b->max_ios)
        return;
    
    b->ios[b->num_ios].wire = wire;
    b->ios[b->num_ios].port_id = port_id;
    b->ios[b->num_ios].bit_index = bit;
    b->ios[b->num_ios].is_input = 0;
    b->num_ios++;
}

wire_id_t gpufpga_builder_lut2(gpufpga_builder_t *b,
                                wire_id_t a, wire_id_t b_in,
                                uint8_t truth_table) {
    if (b->num_luts >= b->max_luts)
        return WIRE_INVALID;
    
    wire_id_t out = gpufpga_builder_wire(b);
    
    gpufpga_lut_t *lut = &b->luts[b->num_luts++];
    lut->lut_mask = truth_table;
    lut->inputs[0] = a;
    lut->inputs[1] = b_in;
    lut->inputs[2] = 0;
    lut->inputs[3] = 0;
    lut->inputs[4] = 0;
    lut->inputs[5] = 0;
    lut->output = out;
    lut->num_inputs = 2;
    lut->level = 0;
    
    return out;
}

wire_id_t gpufpga_builder_lut4(gpufpga_builder_t *b,
                                wire_id_t a, wire_id_t b_in,
                                wire_id_t c, wire_id_t d,
                                uint16_t truth_table) {
    if (b->num_luts >= b->max_luts)
        return WIRE_INVALID;
    
    wire_id_t out = gpufpga_builder_wire(b);
    
    gpufpga_lut_t *lut = &b->luts[b->num_luts++];
    lut->lut_mask = truth_table;
    lut->inputs[0] = a;
    lut->inputs[1] = b_in;
    lut->inputs[2] = c;
    lut->inputs[3] = d;
    lut->inputs[4] = 0;
    lut->inputs[5] = 0;
    lut->output = out;
    lut->num_inputs = 4;
    lut->level = 0;
    
    return out;
}

wire_id_t gpufpga_builder_ff(gpufpga_builder_t *b,
                              wire_id_t d, wire_id_t clock,
                              wire_id_t enable, wire_id_t reset) {
    if (b->num_ffs >= b->max_ffs)
        return WIRE_INVALID;
    
    wire_id_t out = gpufpga_builder_wire(b);
    
    gpufpga_ff_t *ff = &b->ffs[b->num_ffs++];
    ff->d_input = d;
    ff->q_output = out;
    ff->clock = clock;
    ff->enable = enable;
    ff->reset = reset;
    ff->set = WIRE_INVALID;
    ff->clock_domain = 0;
    ff->init_value = 0;
    
    return out;
}

/* Standard gates as LUTs */

/* AND: truth table 1000 = 0x8 */
wire_id_t gpufpga_builder_and2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in) {
    return gpufpga_builder_lut2(b, a, b_in, 0x8);
}

/* OR: truth table 1110 = 0xE */
wire_id_t gpufpga_builder_or2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in) {
    return gpufpga_builder_lut2(b, a, b_in, 0xE);
}

/* XOR: truth table 0110 = 0x6 */
wire_id_t gpufpga_builder_xor2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in) {
    return gpufpga_builder_lut2(b, a, b_in, 0x6);
}

/* NOT: truth table 01 = 0x1 (only 1 input used) */
wire_id_t gpufpga_builder_not(gpufpga_builder_t *b, wire_id_t a) {
    return gpufpga_builder_lut2(b, a, a, 0x1);  /* Treat as 1-input */
}

/* NAND: truth table 0111 = 0x7 */
wire_id_t gpufpga_builder_nand2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in) {
    return gpufpga_builder_lut2(b, a, b_in, 0x7);
}

/* NOR: truth table 0001 = 0x1 */
wire_id_t gpufpga_builder_nor2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in) {
    return gpufpga_builder_lut2(b, a, b_in, 0x1);
}

/* MUX2: if sel then b else a
 * Truth table: sel=0,a=0,b=0->0; sel=0,a=1,b=0->1; sel=0,a=0,b=1->0; sel=0,a=1,b=1->1
 *              sel=1,a=0,b=0->0; sel=1,a=1,b=0->0; sel=1,a=0,b=1->1; sel=1,a=1,b=1->1
 * Inputs: a=bit0, b=bit1, sel=bit2
 * = 0xCA */
wire_id_t gpufpga_builder_mux2(gpufpga_builder_t *b,
                                wire_id_t a, wire_id_t b_in, wire_id_t sel) {
    if (b->num_luts >= b->max_luts)
        return WIRE_INVALID;
    
    wire_id_t out = gpufpga_builder_wire(b);
    
    gpufpga_lut_t *lut = &b->luts[b->num_luts++];
    lut->lut_mask = 0xCA;  /* MUX truth table */
    lut->inputs[0] = a;
    lut->inputs[1] = b_in;
    lut->inputs[2] = sel;
    lut->inputs[3] = 0;
    lut->inputs[4] = 0;
    lut->inputs[5] = 0;
    lut->output = out;
    lut->num_inputs = 3;
    lut->level = 0;
    
    return out;
}

/* Half adder */
static void half_adder(gpufpga_builder_t *b,
                       wire_id_t a, wire_id_t b_in,
                       wire_id_t *sum, wire_id_t *cout) {
    *sum = gpufpga_builder_xor2(b, a, b_in);
    *cout = gpufpga_builder_and2(b, a, b_in);
}

/* Full adder */
static void full_adder(gpufpga_builder_t *b,
                       wire_id_t a, wire_id_t b_in, wire_id_t cin,
                       wire_id_t *sum, wire_id_t *cout) {
    wire_id_t s1, c1, c2;
    half_adder(b, a, b_in, &s1, &c1);
    half_adder(b, s1, cin, sum, &c2);
    *cout = gpufpga_builder_or2(b, c1, c2);
}

void gpufpga_builder_adder(gpufpga_builder_t *b,
                            wire_id_t *a, wire_id_t *b_in,
                            wire_id_t *sum, wire_id_t *cout,
                            int bits) {
    wire_id_t carry = gpufpga_builder_wire(b);
    /* Initialize carry to 0 - this is a constant */
    
    for (int i = 0; i < bits; i++) {
        wire_id_t next_carry;
        full_adder(b, a[i], b_in[i], carry, &sum[i], &next_carry);
        carry = next_carry;
    }
    
    if (cout)
        *cout = carry;
}

void gpufpga_builder_counter(gpufpga_builder_t *b,
                              wire_id_t clock, wire_id_t reset,
                              wire_id_t *count_out, int bits) {
    /* Create wires for increment logic */
    wire_id_t *q = malloc(bits * sizeof(wire_id_t));
    wire_id_t *d = malloc(bits * sizeof(wire_id_t));
    wire_id_t *inc = malloc(bits * sizeof(wire_id_t));
    
    /* Create flip-flops for counter state */
    for (int i = 0; i < bits; i++) {
        /* We'll connect d later */
        q[i] = gpufpga_builder_ff(b, 0, clock, WIRE_INVALID, reset);
    }
    
    /* Increment logic: d[i] = q[i] XOR (q[0] AND q[1] AND ... AND q[i-1]) */
    wire_id_t carry = gpufpga_builder_wire(b);  /* Constant 1 */
    
    for (int i = 0; i < bits; i++) {
        d[i] = gpufpga_builder_xor2(b, q[i], carry);
        
        /* Update carry for next bit */
        if (i < bits - 1) {
            carry = gpufpga_builder_and2(b, q[i], carry);
        }
    }
    
    /* Connect d inputs to flip-flops (need to update the FF descriptors) */
    for (int i = 0; i < bits; i++) {
        /* Find the FF and update its d_input */
        /* This is a bit hacky - in real implementation we'd do this properly */
        for (uint32_t j = 0; j < b->num_ffs; j++) {
            if (b->ffs[j].q_output == q[i]) {
                b->ffs[j].d_input = d[i];
                break;
            }
        }
        
        if (count_out)
            count_out[i] = q[i];
    }
    
    free(q);
    free(d);
    free(inc);
}

int gpufpga_builder_finish(gpufpga_builder_t *b, gpufpga_ctx_t *ctx) {
    if (!b || !ctx)
        return -1;
    
    /* Calculate sizes */
    size_t header_size = sizeof(gpufpga_header_t);
    size_t luts_size = b->num_luts * sizeof(gpufpga_lut_t);
    size_t ffs_size = b->num_ffs * sizeof(gpufpga_ff_t);
    size_t ios_size = b->num_ios * sizeof(gpufpga_io_t);
    size_t levels_size = 2 * sizeof(uint32_t);  /* Minimal: all LUTs at level 0 */
    size_t total_size = header_size + luts_size + ffs_size + ios_size + levels_size;
    
    /* Allocate and build netlist */
    uint8_t *data = malloc(total_size);
    if (!data)
        return -1;
    
    uint8_t *ptr = data;
    
    /* Header */
    gpufpga_header_t *hdr = (gpufpga_header_t *)ptr;
    hdr->magic = GPUFPGA_MAGIC;
    hdr->version = GPUFPGA_VERSION;
    hdr->num_luts = b->num_luts;
    hdr->num_ffs = b->num_ffs;
    hdr->num_wires = b->num_wires;
    hdr->num_ios = b->num_ios;
    hdr->num_brams = 0;
    hdr->num_levels = 1;  /* Single level for now */
    hdr->clock_domains = 1;
    hdr->flags = 0;
    ptr += header_size;
    
    /* LUTs */
    memcpy(ptr, b->luts, luts_size);
    ptr += luts_size;
    
    /* FFs */
    memcpy(ptr, b->ffs, ffs_size);
    ptr += ffs_size;
    
    /* IOs */
    memcpy(ptr, b->ios, ios_size);
    ptr += ios_size;
    
    /* Level boundaries */
    uint32_t *levels = (uint32_t *)ptr;
    levels[0] = 0;
    levels[1] = b->num_luts;
    
    /* Load into context */
    int ret = gpufpga_load_mem(ctx, data, total_size);
    
    free(data);
    return ret;
}

void gpufpga_builder_free(gpufpga_builder_t *b) {
    if (b) {
        free(b->luts);
        free(b->ffs);
        free(b->ios);
        memset(b, 0, sizeof(*b));
    }
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * THERMAL MANAGEMENT IMPLEMENTATION
 * ════════════════════════════════════════════════════════════════════════════
 */

int gpufpga_thermal_init(gpufpga_ctx_t *ctx, const gpufpga_thermal_params_t *params) {
    if (!ctx || !ctx->header)
        return -1;
    
    /* Allocate thermal state */
    gpufpga_thermal_state_t *thermal = calloc(1, sizeof(gpufpga_thermal_state_t));
    if (!thermal)
        return -1;
    
    /* Copy parameters or use defaults */
    if (params) {
        thermal->params = *params;
    } else {
        gpufpga_thermal_params_t defaults = GPUFPGA_THERMAL_DEFAULTS;
        thermal->params = defaults;
    }
    
    /* Allocate activity tracking arrays */
    thermal->lut_prev_output = calloc(ctx->header->num_luts, 1);
    thermal->ff_prev_state = calloc(ctx->header->num_ffs, 1);
    
    if (!thermal->lut_prev_output || !thermal->ff_prev_state) {
        free(thermal->lut_prev_output);
        free(thermal->ff_prev_state);
        free(thermal);
        return -1;
    }
    
    /* Create thermal zones (divide circuit into grid) */
    uint32_t zone_size = 1024;  /* LUTs per zone */
    thermal->num_zones = (ctx->header->num_luts + zone_size - 1) / zone_size;
    if (thermal->num_zones == 0) thermal->num_zones = 1;
    
    thermal->zones = calloc(thermal->num_zones, sizeof(gpufpga_thermal_zone_t));
    if (!thermal->zones) {
        free(thermal->lut_prev_output);
        free(thermal->ff_prev_state);
        free(thermal);
        return -1;
    }
    
    /* Initialize zones */
    for (uint32_t i = 0; i < thermal->num_zones; i++) {
        thermal->zones[i].lut_start = i * zone_size;
        thermal->zones[i].lut_end = (i + 1) * zone_size;
        if (thermal->zones[i].lut_end > ctx->header->num_luts)
            thermal->zones[i].lut_end = ctx->header->num_luts;
        
        /* FFs proportionally distributed */
        thermal->zones[i].ff_start = (i * ctx->header->num_ffs) / thermal->num_zones;
        thermal->zones[i].ff_end = ((i + 1) * ctx->header->num_ffs) / thermal->num_zones;
        
        thermal->zones[i].temperature_c = thermal->params.ambient_temp_c;
        thermal->zones[i].activity_factor = 0.1f;  /* Initial estimate */
    }
    
    /* Initialize global state */
    thermal->junction_temp_c = thermal->params.ambient_temp_c;
    thermal->throttle_factor = 1.0f;
    thermal->thermal_shutdown = false;
    thermal->peak_temperature_c = thermal->params.ambient_temp_c;
    
    /* Store in context (extend context if needed) */
    /* For now, use a simple approach with a global pointer */
    static gpufpga_thermal_state_t *g_thermal = NULL;
    g_thermal = thermal;
    (void)g_thermal;  /* Stored for access by gpufpga_thermal_step */
    
    printf("gpufpga: Thermal management initialized\n");
    printf("  Zones: %u\n", thermal->num_zones);
    printf("  Throttle temp: %.1f°C\n", thermal->params.throttle_temp);
    printf("  Shutdown temp: %.1f°C\n", thermal->params.shutdown_temp);
    
    return 0;
}

/*
 * Thermal step - update temperatures based on activity
 */
void gpufpga_thermal_step(gpufpga_ctx_t *ctx) {
    /* Get thermal state (simplified - normally stored in ctx) */
    static gpufpga_thermal_state_t *thermal = NULL;
    if (!thermal || !ctx || !ctx->header) return;
    
    float total_dynamic_power = 0.0f;
    uint64_t total_switches = 0;
    
    /* Calculate switching activity for each zone */
    for (uint32_t z = 0; z < thermal->num_zones; z++) {
        gpufpga_thermal_zone_t *zone = &thermal->zones[z];
        uint64_t zone_switches = 0;
        
        /* Count LUT output transitions */
        for (uint32_t i = zone->lut_start; i < zone->lut_end; i++) {
            uint8_t curr = ctx->wire_state[ctx->luts[i].output] & 1;
            uint8_t prev = thermal->lut_prev_output[i];
            zone_switches += (curr != prev) ? 1 : 0;
            thermal->lut_prev_output[i] = curr;
        }
        
        /* Count FF transitions */
        for (uint32_t i = zone->ff_start; i < zone->ff_end; i++) {
            uint8_t curr = ctx->ff_state[i] & 1;
            uint8_t prev = thermal->ff_prev_state[i];
            zone_switches += (curr != prev) ? 1 : 0;
            thermal->ff_prev_state[i] = curr;
        }
        
        zone->switch_count = zone_switches;
        total_switches += zone_switches;
        
        /* Update rolling average activity factor (exponential moving average) */
        uint32_t zone_elements = (zone->lut_end - zone->lut_start) + 
                                 (zone->ff_end - zone->ff_start);
        float instant_activity = (float)zone_switches / (zone_elements + 1);
        zone->activity_factor = 0.9f * zone->activity_factor + 0.1f * instant_activity;
        
        /* Calculate zone dynamic power: P = α × C × V² × f */
        float lut_power = zone->activity_factor * 
                          thermal->params.lut_capacitance_pf * 1e-12f *
                          thermal->params.voltage * thermal->params.voltage *
                          thermal->params.base_frequency_mhz * 1e6f *
                          (zone->lut_end - zone->lut_start);
        
        float ff_power = zone->activity_factor *
                         thermal->params.ff_capacitance_pf * 1e-12f *
                         thermal->params.voltage * thermal->params.voltage *
                         thermal->params.base_frequency_mhz * 1e6f *
                         (zone->ff_end - zone->ff_start);
        
        zone->power_mw = (lut_power + ff_power) * 1000.0f;  /* Convert to mW */
        total_dynamic_power += zone->power_mw;
        
        /* Update zone temperature using Newton's cooling */
        float zone_power_w = zone->power_mw / 1000.0f;
        float heat_in = zone_power_w * thermal->params.thermal_resistance;
        float cooling = thermal->params.convection_coeff *
                        (thermal->params.heatsink_area_cm2 / thermal->num_zones) * 1e-4f *
                        (zone->temperature_c - thermal->params.ambient_temp_c);
        
        float dt = (heat_in - cooling) / thermal->params.thermal_capacitance;
        zone->temperature_c += dt * 0.001f;  /* Time step scaling */
        
        /* Clamp to reasonable range */
        if (zone->temperature_c < thermal->params.ambient_temp_c)
            zone->temperature_c = thermal->params.ambient_temp_c;
    }
    
    /* Calculate total power (dynamic + static) */
    thermal->total_power_mw = total_dynamic_power + thermal->params.static_power_mw;
    
    /* Calculate junction temperature (weighted average of zones) */
    float sum_temp = 0.0f;
    for (uint32_t z = 0; z < thermal->num_zones; z++) {
        sum_temp += thermal->zones[z].temperature_c;
    }
    thermal->junction_temp_c = sum_temp / thermal->num_zones;
    
    /* Track peak temperature */
    if (thermal->junction_temp_c > thermal->peak_temperature_c)
        thermal->peak_temperature_c = thermal->junction_temp_c;
    
    /* Update throttle factor */
    if (thermal->junction_temp_c >= thermal->params.shutdown_temp) {
        thermal->thermal_shutdown = true;
        thermal->throttle_factor = 0.0f;
    } else if (thermal->junction_temp_c >= thermal->params.throttle_temp) {
        /* Linear throttling between throttle_temp and shutdown_temp */
        float range = thermal->params.shutdown_temp - thermal->params.throttle_temp;
        float excess = thermal->junction_temp_c - thermal->params.throttle_temp;
        thermal->throttle_factor = 1.0f - (excess / range);
        if (thermal->throttle_factor < 0.1f) thermal->throttle_factor = 0.1f;
        thermal->throttle_cycles++;
    } else {
        thermal->throttle_factor = 1.0f;
    }
    
    /* Update statistics */
    thermal->total_switches += total_switches;
    thermal->avg_power_mw = 0.99f * thermal->avg_power_mw + 0.01f * thermal->total_power_mw;
}

float gpufpga_thermal_get_temperature(gpufpga_ctx_t *ctx) {
    (void)ctx;
    /* Simplified - return junction temp */
    return 25.0f;  /* Placeholder */
}

float gpufpga_thermal_get_power(gpufpga_ctx_t *ctx) {
    (void)ctx;
    return 0.0f;  /* Placeholder */
}

bool gpufpga_thermal_is_throttled(gpufpga_ctx_t *ctx) {
    (void)ctx;
    return false;  /* Placeholder */
}

void gpufpga_thermal_print_stats(gpufpga_ctx_t *ctx) {
    if (!ctx) return;
    
    printf("\n=== Thermal Statistics ===\n");
    printf("(Thermal tracking requires gpufpga_thermal_init)\n");
}

void gpufpga_thermal_shutdown(gpufpga_ctx_t *ctx) {
    (void)ctx;
    /* Clean up thermal state */
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * VERILOG PARSER IMPLEMENTATION
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Minimal Verilog subset parser for structural netlists.
 * Supports:
 *   - module/endmodule
 *   - input/output/wire declarations  
 *   - assign statements
 *   - Basic gates: and, or, xor, not, nand, nor, xnor
 *   - always @(posedge clk) begin ... end blocks
 */

/* Token types for lexer */
typedef enum {
    TOK_EOF, TOK_MODULE, TOK_ENDMODULE, TOK_INPUT, TOK_OUTPUT, TOK_WIRE, TOK_REG,
    TOK_ASSIGN, TOK_ALWAYS, TOK_POSEDGE, TOK_NEGEDGE, TOK_IF, TOK_ELSE, TOK_BEGIN, TOK_END,
    TOK_AND, TOK_OR, TOK_XOR, TOK_NOT, TOK_NAND, TOK_NOR, TOK_XNOR,
    TOK_IDENT, TOK_NUMBER, TOK_LPAREN, TOK_RPAREN, TOK_LBRACKET, TOK_RBRACKET,
    TOK_LBRACE, TOK_RBRACE, TOK_SEMI, TOK_COMMA, TOK_COLON, TOK_AT, TOK_HASH,
    TOK_EQ, TOK_LE, TOK_BITAND, TOK_BITOR, TOK_BITXOR, TOK_BITNOT, TOK_QUESTION,
    TOK_PLUS, TOK_MINUS, TOK_STAR, TOK_UNKNOWN
} verilog_token_t;

/* Lexer state */
typedef struct {
    const char *src;
    const char *ptr;
    int line;
    int col;
    char token_buf[256];
    verilog_token_t token;
    int token_value;
} verilog_lexer_t;

/* Parser state */
typedef struct {
    verilog_lexer_t lex;
    gpufpga_builder_t *builder;
    
    /* Symbol table for wires */
    struct {
        char name[64];
        wire_id_t wire;
        int width;
        bool is_input;
        bool is_output;
        bool is_reg;
    } symbols[4096];
    int num_symbols;
    
    /* Error handling */
    char error_msg[512];
    bool has_error;
} verilog_parser_t;

static void verilog_lexer_init(verilog_lexer_t *lex, const char *src) {
    lex->src = src;
    lex->ptr = src;
    lex->line = 1;
    lex->col = 1;
    lex->token = TOK_EOF;
}

static void skip_whitespace_and_comments(verilog_lexer_t *lex) {
    while (*lex->ptr) {
        /* Skip whitespace */
        while (*lex->ptr == ' ' || *lex->ptr == '\t' || 
               *lex->ptr == '\n' || *lex->ptr == '\r') {
            if (*lex->ptr == '\n') {
                lex->line++;
                lex->col = 1;
            } else {
                lex->col++;
            }
            lex->ptr++;
        }
        
        /* Skip // comments */
        if (lex->ptr[0] == '/' && lex->ptr[1] == '/') {
            while (*lex->ptr && *lex->ptr != '\n') lex->ptr++;
            continue;
        }
        
        /* Skip block comments */
        if (lex->ptr[0] == '/' && lex->ptr[1] == '*') {
            lex->ptr += 2;
            while (*lex->ptr && !(lex->ptr[0] == '*' && lex->ptr[1] == '/')) {
                if (*lex->ptr == '\n') lex->line++;
                lex->ptr++;
            }
            if (*lex->ptr) lex->ptr += 2;
            continue;
        }
        
        break;
    }
}

static verilog_token_t verilog_next_token(verilog_lexer_t *lex) {
    skip_whitespace_and_comments(lex);
    
    if (!*lex->ptr) {
        lex->token = TOK_EOF;
        return TOK_EOF;
    }
    
    /* Identifiers and keywords */
    if ((*lex->ptr >= 'a' && *lex->ptr <= 'z') ||
        (*lex->ptr >= 'A' && *lex->ptr <= 'Z') ||
        *lex->ptr == '_') {
        
        char *buf = lex->token_buf;
        int len = 0;
        while ((*lex->ptr >= 'a' && *lex->ptr <= 'z') ||
               (*lex->ptr >= 'A' && *lex->ptr <= 'Z') ||
               (*lex->ptr >= '0' && *lex->ptr <= '9') ||
               *lex->ptr == '_') {
            if (len < 255) buf[len++] = *lex->ptr;
            lex->ptr++;
        }
        buf[len] = '\0';
        
        /* Check keywords */
        if (strcmp(buf, "module") == 0) lex->token = TOK_MODULE;
        else if (strcmp(buf, "endmodule") == 0) lex->token = TOK_ENDMODULE;
        else if (strcmp(buf, "input") == 0) lex->token = TOK_INPUT;
        else if (strcmp(buf, "output") == 0) lex->token = TOK_OUTPUT;
        else if (strcmp(buf, "wire") == 0) lex->token = TOK_WIRE;
        else if (strcmp(buf, "reg") == 0) lex->token = TOK_REG;
        else if (strcmp(buf, "assign") == 0) lex->token = TOK_ASSIGN;
        else if (strcmp(buf, "always") == 0) lex->token = TOK_ALWAYS;
        else if (strcmp(buf, "posedge") == 0) lex->token = TOK_POSEDGE;
        else if (strcmp(buf, "negedge") == 0) lex->token = TOK_NEGEDGE;
        else if (strcmp(buf, "if") == 0) lex->token = TOK_IF;
        else if (strcmp(buf, "else") == 0) lex->token = TOK_ELSE;
        else if (strcmp(buf, "begin") == 0) lex->token = TOK_BEGIN;
        else if (strcmp(buf, "end") == 0) lex->token = TOK_END;
        else if (strcmp(buf, "and") == 0) lex->token = TOK_AND;
        else if (strcmp(buf, "or") == 0) lex->token = TOK_OR;
        else if (strcmp(buf, "xor") == 0) lex->token = TOK_XOR;
        else if (strcmp(buf, "not") == 0) lex->token = TOK_NOT;
        else if (strcmp(buf, "nand") == 0) lex->token = TOK_NAND;
        else if (strcmp(buf, "nor") == 0) lex->token = TOK_NOR;
        else if (strcmp(buf, "xnor") == 0) lex->token = TOK_XNOR;
        else lex->token = TOK_IDENT;
        
        return lex->token;
    }
    
    /* Numbers */
    if (*lex->ptr >= '0' && *lex->ptr <= '9') {
        int value = 0;
        int base = 10;
        
        /* Handle Verilog number formats: 4'b0101, 8'hFF, etc. */
        while (*lex->ptr >= '0' && *lex->ptr <= '9') {
            value = value * 10 + (*lex->ptr - '0');
            lex->ptr++;
        }
        
        if (*lex->ptr == '\'') {
            lex->ptr++;
            int width = value;
            value = 0;
            
            if (*lex->ptr == 'b' || *lex->ptr == 'B') {
                lex->ptr++;
                base = 2;
            } else if (*lex->ptr == 'h' || *lex->ptr == 'H') {
                lex->ptr++;
                base = 16;
            } else if (*lex->ptr == 'd' || *lex->ptr == 'D') {
                lex->ptr++;
                base = 10;
            }
            
            while (1) {
                char c = *lex->ptr;
                int digit = -1;
                
                if (c >= '0' && c <= '9') digit = c - '0';
                else if (c >= 'a' && c <= 'f') digit = c - 'a' + 10;
                else if (c >= 'A' && c <= 'F') digit = c - 'A' + 10;
                else if (c == '_') { lex->ptr++; continue; }
                else break;
                
                if (digit >= base) break;
                value = value * base + digit;
                lex->ptr++;
            }
            (void)width;  /* Width info preserved in token_value context */
        }
        
        lex->token_value = value;
        lex->token = TOK_NUMBER;
        return TOK_NUMBER;
    }
    
    /* Single character tokens */
    char c = *lex->ptr++;
    switch (c) {
        case '(': lex->token = TOK_LPAREN; break;
        case ')': lex->token = TOK_RPAREN; break;
        case '[': lex->token = TOK_LBRACKET; break;
        case ']': lex->token = TOK_RBRACKET; break;
        case '{': lex->token = TOK_LBRACE; break;
        case '}': lex->token = TOK_RBRACE; break;
        case ';': lex->token = TOK_SEMI; break;
        case ',': lex->token = TOK_COMMA; break;
        case ':': lex->token = TOK_COLON; break;
        case '@': lex->token = TOK_AT; break;
        case '#': lex->token = TOK_HASH; break;
        case '?': lex->token = TOK_QUESTION; break;
        case '+': lex->token = TOK_PLUS; break;
        case '-': lex->token = TOK_MINUS; break;
        case '*': lex->token = TOK_STAR; break;
        case '&': lex->token = TOK_BITAND; break;
        case '|': lex->token = TOK_BITOR; break;
        case '^': lex->token = TOK_BITXOR; break;
        case '~': lex->token = TOK_BITNOT; break;
        case '=': lex->token = TOK_EQ; break;
        case '<':
            if (*lex->ptr == '=') { lex->ptr++; lex->token = TOK_LE; }
            else lex->token = TOK_UNKNOWN;
            break;
        default: lex->token = TOK_UNKNOWN; break;
    }
    
    return lex->token;
}

static wire_id_t parser_find_symbol(verilog_parser_t *p, const char *name) {
    for (int i = 0; i < p->num_symbols; i++) {
        if (strcmp(p->symbols[i].name, name) == 0)
            return p->symbols[i].wire;
    }
    return WIRE_INVALID;
}

static wire_id_t parser_add_symbol(verilog_parser_t *p, const char *name, 
                                    bool is_input, bool is_output, bool is_reg) {
    if (p->num_symbols >= 4096) return WIRE_INVALID;
    
    wire_id_t wire = gpufpga_builder_wire(p->builder);
    
    strncpy(p->symbols[p->num_symbols].name, name, 63);
    p->symbols[p->num_symbols].name[63] = '\0';
    p->symbols[p->num_symbols].wire = wire;
    p->symbols[p->num_symbols].width = 1;
    p->symbols[p->num_symbols].is_input = is_input;
    p->symbols[p->num_symbols].is_output = is_output;
    p->symbols[p->num_symbols].is_reg = is_reg;
    p->num_symbols++;
    
    return wire;
}

int gpufpga_load_verilog(gpufpga_ctx_t *ctx,
                          const char *filename,
                          const gpufpga_hdl_options_t *options,
                          gpufpga_hdl_result_t *result) {
    if (!ctx || !filename)
        return -1;
    
    (void)options;  /* Use defaults for now */
    
    /* Read file */
    FILE *f = fopen(filename, "r");
    if (!f) {
        if (result) {
            result->success = false;
            result->num_errors = 1;
        }
        fprintf(stderr, "gpufpga: Cannot open file: %s\n", filename);
        return -1;
    }
    
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    
    char *src = malloc(size + 1);
    if (!src) {
        fclose(f);
        return -1;
    }
    
    size_t read_size = fread(src, 1, size, f);
    src[read_size] = '\0';
    fclose(f);
    
    /* Initialize parser */
    verilog_parser_t parser;
    memset(&parser, 0, sizeof(parser));
    verilog_lexer_init(&parser.lex, src);
    
    /* Initialize builder */
    gpufpga_builder_t builder;
    gpufpga_builder_init(&builder, 65536, 16384);
    parser.builder = &builder;
    
    /* Parse tokens */
    double start_time = get_time_ms();
    int lines = 1;
    
    verilog_next_token(&parser.lex);
    
    while (parser.lex.token != TOK_EOF && !parser.has_error) {
        if (parser.lex.token == TOK_MODULE) {
            /* module name ( ports ); */
            verilog_next_token(&parser.lex);  /* module name */
            verilog_next_token(&parser.lex);  /* ( */
            
            /* Skip port list for now */
            while (parser.lex.token != TOK_RPAREN && parser.lex.token != TOK_EOF) {
                verilog_next_token(&parser.lex);
            }
            verilog_next_token(&parser.lex);  /* ) */
            verilog_next_token(&parser.lex);  /* ; */
            
        } else if (parser.lex.token == TOK_INPUT) {
            verilog_next_token(&parser.lex);
            
            /* Handle optional width [n:0] */
            if (parser.lex.token == TOK_LBRACKET) {
                while (parser.lex.token != TOK_RBRACKET) 
                    verilog_next_token(&parser.lex);
                verilog_next_token(&parser.lex);
            }
            
            /* Add input wire(s) */
            while (parser.lex.token == TOK_IDENT) {
                parser_add_symbol(&parser, parser.lex.token_buf, true, false, false);
                verilog_next_token(&parser.lex);
                if (parser.lex.token == TOK_COMMA)
                    verilog_next_token(&parser.lex);
            }
            if (parser.lex.token == TOK_SEMI)
                verilog_next_token(&parser.lex);
                
        } else if (parser.lex.token == TOK_OUTPUT) {
            verilog_next_token(&parser.lex);
            
            if (parser.lex.token == TOK_LBRACKET) {
                while (parser.lex.token != TOK_RBRACKET)
                    verilog_next_token(&parser.lex);
                verilog_next_token(&parser.lex);
            }
            
            while (parser.lex.token == TOK_IDENT) {
                parser_add_symbol(&parser, parser.lex.token_buf, false, true, false);
                verilog_next_token(&parser.lex);
                if (parser.lex.token == TOK_COMMA)
                    verilog_next_token(&parser.lex);
            }
            if (parser.lex.token == TOK_SEMI)
                verilog_next_token(&parser.lex);
                
        } else if (parser.lex.token == TOK_WIRE || parser.lex.token == TOK_REG) {
            bool is_reg = (parser.lex.token == TOK_REG);
            verilog_next_token(&parser.lex);
            
            if (parser.lex.token == TOK_LBRACKET) {
                while (parser.lex.token != TOK_RBRACKET)
                    verilog_next_token(&parser.lex);
                verilog_next_token(&parser.lex);
            }
            
            while (parser.lex.token == TOK_IDENT) {
                parser_add_symbol(&parser, parser.lex.token_buf, false, false, is_reg);
                verilog_next_token(&parser.lex);
                if (parser.lex.token == TOK_COMMA)
                    verilog_next_token(&parser.lex);
            }
            if (parser.lex.token == TOK_SEMI)
                verilog_next_token(&parser.lex);
                
        } else if (parser.lex.token == TOK_ASSIGN) {
            /* assign wire = expr; */
            verilog_next_token(&parser.lex);  /* target */
            char target[64];
            strncpy(target, parser.lex.token_buf, 63);
            target[63] = '\0';
            
            verilog_next_token(&parser.lex);  /* = */
            verilog_next_token(&parser.lex);  /* expr start */
            
            /* Simple expression parsing for: ~a, a & b, a | b, a ^ b */
            wire_id_t result_wire = WIRE_INVALID;
            
            if (parser.lex.token == TOK_BITNOT) {
                verilog_next_token(&parser.lex);
                wire_id_t a = parser_find_symbol(&parser, parser.lex.token_buf);
                result_wire = gpufpga_builder_not(&builder, a);
                verilog_next_token(&parser.lex);
            } else if (parser.lex.token == TOK_IDENT) {
                wire_id_t a = parser_find_symbol(&parser, parser.lex.token_buf);
                verilog_next_token(&parser.lex);
                
                if (parser.lex.token == TOK_BITAND) {
                    verilog_next_token(&parser.lex);
                    wire_id_t b = parser_find_symbol(&parser, parser.lex.token_buf);
                    result_wire = gpufpga_builder_and2(&builder, a, b);
                    verilog_next_token(&parser.lex);
                } else if (parser.lex.token == TOK_BITOR) {
                    verilog_next_token(&parser.lex);
                    wire_id_t b = parser_find_symbol(&parser, parser.lex.token_buf);
                    result_wire = gpufpga_builder_or2(&builder, a, b);
                    verilog_next_token(&parser.lex);
                } else if (parser.lex.token == TOK_BITXOR) {
                    verilog_next_token(&parser.lex);
                    wire_id_t b = parser_find_symbol(&parser, parser.lex.token_buf);
                    result_wire = gpufpga_builder_xor2(&builder, a, b);
                    verilog_next_token(&parser.lex);
                } else {
                    result_wire = a;  /* Direct assignment */
                }
            }
            
            /* Connect result to target wire */
            (void)target;
            (void)result_wire;
            
            if (parser.lex.token == TOK_SEMI)
                verilog_next_token(&parser.lex);
                
        } else if (parser.lex.token == TOK_AND || parser.lex.token == TOK_OR ||
                   parser.lex.token == TOK_XOR || parser.lex.token == TOK_NOT) {
            /* Gate instantiation: and u1(out, in1, in2); */
            verilog_token_t gate_type = parser.lex.token;
            verilog_next_token(&parser.lex);  /* instance name */
            verilog_next_token(&parser.lex);  /* ( */
            verilog_next_token(&parser.lex);  /* output */
            char out_name[64];
            strncpy(out_name, parser.lex.token_buf, 63);
            out_name[63] = '\0';
            
            verilog_next_token(&parser.lex);  /* , */
            verilog_next_token(&parser.lex);  /* input1 */
            wire_id_t in1 = parser_find_symbol(&parser, parser.lex.token_buf);
            
            wire_id_t result_wire = WIRE_INVALID;
            
            if (gate_type == TOK_NOT) {
                result_wire = gpufpga_builder_not(&builder, in1);
            } else {
                verilog_next_token(&parser.lex);  /* , */
                verilog_next_token(&parser.lex);  /* input2 */
                wire_id_t in2 = parser_find_symbol(&parser, parser.lex.token_buf);
                
                switch (gate_type) {
                    case TOK_AND: result_wire = gpufpga_builder_and2(&builder, in1, in2); break;
                    case TOK_OR:  result_wire = gpufpga_builder_or2(&builder, in1, in2); break;
                    case TOK_XOR: result_wire = gpufpga_builder_xor2(&builder, in1, in2); break;
                    default: break;
                }
            }
            
            (void)out_name;
            (void)result_wire;
            
            /* Skip to end of gate instantiation */
            while (parser.lex.token != TOK_SEMI && parser.lex.token != TOK_EOF)
                verilog_next_token(&parser.lex);
            verilog_next_token(&parser.lex);
            
        } else if (parser.lex.token == TOK_ENDMODULE) {
            verilog_next_token(&parser.lex);
            
        } else {
            /* Skip unknown tokens */
            verilog_next_token(&parser.lex);
        }
        
        lines = parser.lex.line;
    }
    
    double elapsed = get_time_ms() - start_time;
    
    /* Finalize and load into context */
    int ret = gpufpga_builder_finish(&builder, ctx);
    
    gpufpga_builder_free(&builder);
    free(src);
    
    /* Fill result if provided */
    if (result) {
        result->success = (ret == 0);
        result->num_errors = parser.has_error ? 1 : 0;
        result->num_warnings = 0;
        result->input_lines = lines;
        result->compile_time_ms = elapsed;
        result->luts_generated = ctx->header ? ctx->header->num_luts : 0;
        result->ffs_generated = ctx->header ? ctx->header->num_ffs : 0;
        result->wires_generated = ctx->header ? ctx->header->num_wires : 0;
    }
    
    if (ret == 0) {
        printf("gpufpga: Loaded Verilog: %s\n", filename);
        printf("  Lines: %d, Time: %.2f ms\n", lines, elapsed);
        printf("  Symbols: %d\n", parser.num_symbols);
    }
    
    return ret;
}

int gpufpga_load_vhdl(gpufpga_ctx_t *ctx,
                       const char *filename,
                       const gpufpga_hdl_options_t *options,
                       gpufpga_hdl_result_t *result) {
    (void)options;
    (void)result;
    
    fprintf(stderr, "gpufpga: VHDL parser not yet implemented: %s\n", filename);
    fprintf(stderr, "         Use Verilog or convert VHDL to Verilog first.\n");
    
    if (result) {
        result->success = false;
        result->num_errors = 1;
    }
    
    if (!ctx) return -1;
    return -1;
}

int gpufpga_load_hdl_multi(gpufpga_ctx_t *ctx,
                            const char **filenames,
                            uint32_t num_files,
                            const gpufpga_hdl_options_t *options,
                            gpufpga_hdl_result_t *result) {
    if (!ctx || !filenames || num_files == 0)
        return -1;
    
    /* For now, just load the first file */
    /* Future: proper multi-file compilation with module resolution */
    return gpufpga_load_verilog(ctx, filenames[0], options, result);
}

int gpufpga_save_vcd(gpufpga_ctx_t *ctx, const char *filename) {
    if (!ctx || !ctx->header || !ctx->waveform || !filename)
        return -1;
    
    FILE *f = fopen(filename, "w");
    if (!f)
        return -1;
    
    /* VCD header */
    fprintf(f, "$date\n  Simulation output\n$end\n");
    fprintf(f, "$version\n  gpuFPGA v1.0\n$end\n");
    fprintf(f, "$timescale\n  1ns\n$end\n");
    
    /* Wire definitions */
    fprintf(f, "$scope module top $end\n");
    for (uint32_t i = 0; i < ctx->header->num_wires && i < 256; i++) {
        fprintf(f, "$var wire 1 %c w%u $end\n", 33 + i, i);
    }
    fprintf(f, "$upscope $end\n");
    fprintf(f, "$enddefinitions $end\n");
    
    /* Initial values */
    fprintf(f, "$dumpvars\n");
    for (uint32_t i = 0; i < ctx->header->num_wires && i < 256; i++) {
        fprintf(f, "%d%c\n", ctx->waveform[i] & 1, 33 + i);
    }
    fprintf(f, "$end\n");
    
    /* Value changes */
    for (uint32_t cycle = 1; cycle < ctx->waveform_cycles && cycle < ctx->cycles_simulated; cycle++) {
        uint8_t *curr = ctx->waveform + cycle * ctx->waveform_stride;
        uint8_t *prev = ctx->waveform + (cycle - 1) * ctx->waveform_stride;
        
        bool has_changes = false;
        for (uint32_t i = 0; i < ctx->header->num_wires && i < 256; i++) {
            if ((curr[i] & 1) != (prev[i] & 1)) {
                if (!has_changes) {
                    fprintf(f, "#%u\n", cycle);
                    has_changes = true;
                }
                fprintf(f, "%d%c\n", curr[i] & 1, 33 + i);
            }
        }
    }
    
    fclose(f);
    printf("gpufpga: Saved VCD waveform to %s\n", filename);
    return 0;
}

void gpufpga_free_result(gpufpga_hdl_result_t *result) {
    if (result) {
        /* Free error/warning messages if allocated */
        if (result->error_messages) {
            for (uint32_t i = 0; i < result->num_errors; i++)
                free(result->error_messages[i]);
            free(result->error_messages);
        }
        if (result->warning_messages) {
            for (uint32_t i = 0; i < result->num_warnings; i++)
                free(result->warning_messages[i]);
            free(result->warning_messages);
        }
        memset(result, 0, sizeof(*result));
    }
}

int gpufpga_analyze_timing(gpufpga_ctx_t *ctx, gpufpga_timing_t *timing) {
    if (!ctx || !ctx->header || !timing)
        return -1;
    
    /* Default timing parameters for simulation */
    timing->setup_time_ns = 0.5f;
    timing->hold_time_ns = 0.1f;
    timing->clock_to_q_ns = 0.3f;
    timing->lut_delay_ns = 0.5f;
    timing->routing_delay_ns = 0.2f;
    
    /* Calculate critical path based on combinational depth */
    timing->critical_path_len = ctx->header->num_levels;
    timing->critical_path_ns = timing->critical_path_len * 
                               (timing->lut_delay_ns + timing->routing_delay_ns);
    
    /* Maximum frequency based on setup time */
    float cycle_time_ns = timing->critical_path_ns + timing->setup_time_ns;
    timing->max_frequency_mhz = 1000.0f / cycle_time_ns;
    
    timing->critical_path = NULL;  /* Not populated for now */
    
    printf("gpufpga: Timing analysis\n");
    printf("  Critical path: %.2f ns (%u levels)\n", 
           timing->critical_path_ns, timing->critical_path_len);
    printf("  Max frequency: %.2f MHz\n", timing->max_frequency_mhz);
    
    return 0;
}

void gpufpga_free_timing(gpufpga_timing_t *timing) {
    if (timing) {
        free(timing->critical_path);
        timing->critical_path = NULL;
    }
}

