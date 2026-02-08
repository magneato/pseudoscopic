/*
 * gpufpga.c - GPU-Accelerated FPGA Emulator Implementation
 *
 * Branchless FPGA simulation via near-memory computing.
 * Every LUT is a table lookup. Every FF is a conditional move.
 * No branches. Pure parallel evaluation.
 *
 * Copyright (C) 2025 Neural Splines LLC
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
