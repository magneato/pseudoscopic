/*
 * gpuFPGA - FPGA Emulation via Branchless GPU Computing
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE REVELATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * FPGAs don't execute instructions. They ARE the circuit.
 * 
 * An FPGA is:
 *   - Thousands of Look-Up Tables (LUTs) - truth tables in silicon
 *   - Thousands of Flip-Flops (FFs) - 1-bit state elements
 *   - A routing network - wires connecting everything
 *   - Clock distribution - synchronizes state updates
 *
 * Every LUT evaluates SIMULTANEOUSLY on every clock cycle.
 * Every flip-flop captures its input SIMULTANEOUSLY.
 * There are NO BRANCHES in hardware - just parallel evaluation.
 *
 * This is PERFECT for GPU emulation:
 *   - 1 CUDA thread per LUT (or per small group)
 *   - Shared memory for routing/interconnect
 *   - __syncthreads() for clock edges
 *   - Branchless evaluation via table lookup
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HOW FPGAs ACTUALLY WORK
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 1. LOOK-UP TABLES (LUTs)
 * ────────────────────────
 * A k-input LUT implements ANY k-input Boolean function.
 * It's literally a 2^k entry truth table stored in SRAM.
 *
 *   4-input LUT (most common):
 *   ┌─────────────────────────────┐
 *   │  Inputs: A, B, C, D         │
 *   │  Address = {D,C,B,A} (4-bit)│
 *   │  Output = SRAM[Address]     │
 *   └─────────────────────────────┘
 *
 *   Example: 4-input AND gate
 *   Address: 0000 0001 0010 0011 0100 0101 0110 0111 1000 1001 1010 1011 1100 1101 1110 1111
 *   Output:    0    0    0    0    0    0    0    0    0    0    0    0    0    0    0    1
 *
 *   The LUT evaluation is BRANCHLESS:
 *     output = lut_sram[(d << 3) | (c << 2) | (b << 1) | a];
 *
 *   No if-else. No branches. Just array indexing.
 *
 * 2. FLIP-FLOPS (FFs)
 * ───────────────────
 * A flip-flop captures its D input on the rising edge of clock.
 *
 *   ┌─────────────────────────────┐
 *   │     D ──┬──► Q              │
 *   │         │                   │
 *   │   CLK ──┘                   │
 *   └─────────────────────────────┘
 *
 *   Branchless update (using clock enable):
 *     q_next = (clock_edge & d) | (~clock_edge & q);
 *
 *   Or with conditional move intrinsic:
 *     q_next = clock_edge ? d : q;  // Compiler optimizes to CMOV
 *
 * 3. ROUTING NETWORK
 * ──────────────────
 * Wires connecting LUT outputs to LUT/FF inputs.
 * In hardware: physical metal traces.
 * In emulation: memory copies or index lookups.
 *
 *   wire[dest] = wire[source];
 *
 *   This is just memory movement - no computation, no branches.
 *
 * 4. CLOCK DOMAINS
 * ────────────────
 * Most designs have one or few clocks.
 * All FFs in a domain update simultaneously.
 *
 *   Emulation cycle:
 *     1. Evaluate all combinational logic (LUTs) - PARALLEL
 *     2. Propagate signals through routing - PARALLEL
 *     3. Update all flip-flops on clock edge - PARALLEL
 *     4. Repeat
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * GPU MAPPING
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The key insight: FPGA evaluation maps directly to GPU execution model.
 *
 *   FPGA Concept          │  GPU Equivalent
 *   ──────────────────────┼──────────────────────────────────
 *   LUT                   │  Thread (or warp for larger LUTs)
 *   LUT SRAM              │  Shared memory or constant memory
 *   Flip-flop             │  Register or shared memory location
 *   Routing network       │  Shared memory reads/writes
 *   Clock edge            │  __syncthreads() barrier
 *   Clock cycle           │  Kernel iteration
 *   Chip                  │  Thread block (or multiple blocks)
 *
 * Memory hierarchy mapping:
 *   ┌────────────────────────────────────────────────────────────────────┐
 *   │  GPU Global Memory (VRAM via pseudoscopic)                        │
 *   │    - Netlist description                                          │
 *   │    - LUT contents (truth tables)                                  │
 *   │    - Testbench data (inputs/expected outputs)                     │
 *   │    - Waveform capture                                             │
 *   ├────────────────────────────────────────────────────────────────────┤
 *   │  Shared Memory (per block, fast)                                  │
 *   │    - Wire state (current signal values)                           │
 *   │    - FF state (registered values)                                 │
 *   │    - Small LUT tables                                             │
 *   ├────────────────────────────────────────────────────────────────────┤
 *   │  Registers (per thread, fastest)                                  │
 *   │    - Local LUT inputs/outputs                                     │
 *   │    - Loop counters                                                │
 *   └────────────────────────────────────────────────────────────────────┘
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * BRANCHLESS EVALUATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * GPUs HATE branches. Divergent warps serialize execution.
 * But FPGA logic is naturally branchless!
 *
 * BRANCHLESS LUT EVALUATION:
 * ──────────────────────────
 *   // 4-input LUT: 16-entry truth table
 *   __device__ uint8_t eval_lut4(uint16_t lut_mask, uint8_t inputs) {
 *       // inputs = 4 bits packed: {d, c, b, a}
 *       // lut_mask = 16 bits, one per input combination
 *       return (lut_mask >> inputs) & 1;
 *   }
 *
 *   // 6-input LUT: 64-entry truth table
 *   __device__ uint8_t eval_lut6(uint64_t lut_mask, uint8_t inputs) {
 *       return (lut_mask >> inputs) & 1;
 *   }
 *
 *   No branches! Just bit shifts and masks.
 *
 * BRANCHLESS MUX (routing):
 * ─────────────────────────
 *   // 2:1 mux without branches
 *   __device__ uint8_t mux2(uint8_t a, uint8_t b, uint8_t sel) {
 *       return (a & ~sel) | (b & sel);  // sel must be 0 or 1
 *   }
 *
 *   // 4:1 mux without branches
 *   __device__ uint8_t mux4(uint8_t *inputs, uint8_t sel) {
 *       // Use bit manipulation to select
 *       uint8_t mask0 = -(sel == 0);
 *       uint8_t mask1 = -(sel == 1);
 *       uint8_t mask2 = -(sel == 2);
 *       uint8_t mask3 = -(sel == 3);
 *       return (inputs[0] & mask0) | (inputs[1] & mask1) |
 *              (inputs[2] & mask2) | (inputs[3] & mask3);
 *   }
 *
 * BRANCHLESS FF UPDATE:
 * ─────────────────────
 *   // D flip-flop with enable and reset
 *   __device__ uint8_t ff_update(uint8_t q, uint8_t d, 
 *                                 uint8_t en, uint8_t rst) {
 *       // rst=1: output 0
 *       // en=1, rst=0: output d
 *       // en=0, rst=0: output q (hold)
 *       uint8_t not_rst = ~rst & 1;
 *       uint8_t update = en & not_rst;
 *       return (d & update) | (q & ~update & not_rst);
 *   }
 *
 *   No if-else anywhere!
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * NETLIST REPRESENTATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * A netlist describes the circuit structure:
 *   - List of LUTs with their truth tables
 *   - List of FFs with their connections
 *   - Routing: which signals connect where
 *
 * Compact representation for GPU:
 *
 *   // LUT descriptor (fits in 16 bytes)
 *   typedef struct {
 *       uint64_t lut_mask;      // Truth table (up to 6 inputs)
 *       uint16_t input_ids[4];  // Wire IDs for inputs (or 6 for LUT6)
 *       uint16_t output_id;     // Wire ID for output
 *       uint8_t  num_inputs;    // 2-6 typically
 *       uint8_t  flags;         // Carry chain, etc.
 *   } lut_desc_t;
 *
 *   // FF descriptor (fits in 8 bytes)
 *   typedef struct {
 *       uint16_t d_input;       // Wire ID for D input
 *       uint16_t q_output;      // Wire ID for Q output
 *       uint16_t clock_id;      // Clock domain
 *       uint8_t  has_enable;    // Enable input present
 *       uint8_t  has_reset;     // Reset input present
 *   } ff_desc_t;
 *
 *   // Wire state: simple array
 *   uint8_t wire_state[MAX_WIRES];  // 0 or 1 per wire
 *
 * Memory layout in VRAM:
 *   ┌─────────────────────────────────────────────────────────────────┐
 *   │  Header (circuit metadata)                                      │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  LUT descriptors [num_luts]                                     │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  FF descriptors [num_ffs]                                       │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  Wire state (current) [num_wires]                               │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  Wire state (next) [num_wires]                                  │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  FF state [num_ffs]                                             │
 *   ├─────────────────────────────────────────────────────────────────┤
 *   │  Waveform capture buffer [cycles × trace_wires]                 │
 *   └─────────────────────────────────────────────────────────────────┘
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE SIMULATION KERNEL
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * The core simulation loop:
 *
 *   __global__ void fpga_simulate(
 *       lut_desc_t *luts, int num_luts,
 *       ff_desc_t *ffs, int num_ffs,
 *       uint8_t *wire_state,
 *       uint8_t *wire_next,
 *       uint8_t *ff_state,
 *       int clock_edge)
 *   {
 *       // Each thread handles some LUTs
 *       int tid = blockIdx.x * blockDim.x + threadIdx.x;
 *       
 *       // ═══ PHASE 1: Evaluate combinational logic ═══
 *       for (int i = tid; i < num_luts; i += gridDim.x * blockDim.x) {
 *           lut_desc_t *lut = &luts[i];
 *           
 *           // Gather inputs (BRANCHLESS)
 *           uint8_t inputs = 0;
 *           inputs |= wire_state[lut->input_ids[0]] << 0;
 *           inputs |= wire_state[lut->input_ids[1]] << 1;
 *           inputs |= wire_state[lut->input_ids[2]] << 2;
 *           inputs |= wire_state[lut->input_ids[3]] << 3;
 *           
 *           // Evaluate LUT (BRANCHLESS - just bit extraction)
 *           uint8_t output = (lut->lut_mask >> inputs) & 1;
 *           
 *           // Write to next state
 *           wire_next[lut->output_id] = output;
 *       }
 *       
 *       __syncthreads();  // Barrier: all combinational logic done
 *       
 *       // ═══ PHASE 2: Copy next state to current ═══
 *       // (Could be optimized with double-buffering)
 *       for (int i = tid; i < num_wires; i += gridDim.x * blockDim.x) {
 *           wire_state[i] = wire_next[i];
 *       }
 *       
 *       __syncthreads();  // Barrier: routing propagation done
 *       
 *       // ═══ PHASE 3: Update flip-flops on clock edge ═══
 *       if (clock_edge) {
 *           for (int i = tid; i < num_ffs; i += gridDim.x * blockDim.x) {
 *               ff_desc_t *ff = &ffs[i];
 *               
 *               // BRANCHLESS FF update
 *               uint8_t d = wire_state[ff->d_input];
 *               ff_state[i] = d;
 *               wire_state[ff->q_output] = d;
 *           }
 *       }
 *   }
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * HANDLING COMBINATIONAL LOOPS (LEVELIZATION)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Problem: Some combinational paths may need multiple iterations to settle.
 *
 *   A → LUT1 → B → LUT2 → C → LUT3 → D
 *
 * If we evaluate in parallel, LUT3 might use stale value of C.
 *
 * Solution: LEVELIZATION
 *   1. Topologically sort LUTs by combinational depth
 *   2. Evaluate level-by-level with barriers
 *   3. Or iterate until stable (event-driven)
 *
 *   Level 0: LUTs with only FF/input sources
 *   Level 1: LUTs depending on Level 0
 *   Level 2: LUTs depending on Level 1
 *   ...
 *
 *   __global__ void fpga_simulate_leveled(
 *       lut_desc_t *luts,
 *       int *level_starts,  // level_starts[i] = first LUT in level i
 *       int num_levels,
 *       uint8_t *wire_state)
 *   {
 *       for (int level = 0; level < num_levels; level++) {
 *           int start = level_starts[level];
 *           int end = level_starts[level + 1];
 *           
 *           // Evaluate all LUTs at this level IN PARALLEL
 *           for (int i = start + tid; i < end; i += stride) {
 *               evaluate_lut(&luts[i], wire_state);
 *           }
 *           
 *           __syncthreads();  // Barrier between levels
 *       }
 *   }
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * EVENT-DRIVEN SIMULATION (ADVANCED)
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * For sparse activity (most signals don't change each cycle):
 *
 *   1. Maintain "changed" bitmap for wires
 *   2. Only evaluate LUTs whose inputs changed
 *   3. Mark outputs as changed if they differ
 *   4. Iterate until no changes (stable)
 *
 * This is how production simulators (Verilator, VCS) work.
 * Tricky on GPU due to dynamic parallelism, but doable.
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * COMPARISON TO EXISTING FPGA EMULATORS
 * ═══════════════════════════════════════════════════════════════════════════
 *
 *   Tool           │  Method              │  Speed          │  Use Case
 *   ───────────────┼──────────────────────┼─────────────────┼───────────────
 *   Icarus Verilog │  Interpreted         │  Slow (kHz)     │  Functional
 *   Verilator      │  Compiled to C++     │  Fast (MHz)     │  Verification
 *   VCS/ModelSim   │  Commercial, mixed   │  Medium         │  Production
 *   FPGA Proto     │  Actual FPGA         │  Real-time      │  HW debug
 *   gpuFPGA        │  GPU parallel        │  Fast (MHz?)    │  Novel!
 *
 * gpuFPGA advantages:
 *   - Massive parallelism (10,000+ threads)
 *   - HBM2 bandwidth for wire state (700+ GB/s)
 *   - Can emulate designs larger than single FPGA
 *   - Integrates with pseudoscopic for huge designs in VRAM
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * NEAR-MEMORY INTEGRATION
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * With pseudoscopic, the entire circuit state lives in VRAM:
 *
 *   1. Load netlist from file into VRAM (via BAR1 mmap)
 *   2. Initialize wire/FF state
 *   3. GPU kernel runs simulation cycles
 *   4. CPU can inspect state via BAR1 at any time
 *   5. Waveform capture stays in VRAM (no copy out)
 *
 * For HUGE designs (millions of LUTs):
 *   - Tile the design across multiple kernel launches
 *   - Use tiled evaluation from nearmem_tile
 *   - Stream waveform data to disk via pseudoscopic
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * WHAT THIS ENABLES
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * 1. FAST RTL SIMULATION
 *    - Compile Verilog/VHDL to netlist
 *    - Load netlist into gpuFPGA
 *    - Simulate at millions of cycles per second
 *
 * 2. FPGA DEVELOPMENT WITHOUT FPGA
 *    - Debug designs before hardware arrives
 *    - Run regression tests in cloud (GPU instances)
 *    - No expensive FPGA licenses
 *
 * 3. SOFT PROCESSOR ON GPU
 *    - Implement RISC-V or ARM in Verilog
 *    - Simulate on GPU
 *    - Run software on emulated processor!
 *    - GPU → FPGA → CPU (triple indirection!)
 *
 * 4. NEURAL NETWORK ACCELERATOR EMULATION
 *    - Design custom neural network accelerator in RTL
 *    - Simulate on GPU
 *    - Verify correctness before tapeout
 *
 * 5. THE RECURSIVE POSSIBILITY
 *    - Emulate FPGA on GPU
 *    - FPGA design implements GPU
 *    - ??? (turtles all the way down)
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * IMPLEMENTATION ROADMAP
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * Phase 1: Basic LUT/FF Simulation (this file)
 *   [x] Branchless LUT evaluation
 *   [x] Branchless FF update
 *   [x] Netlist representation
 *   [ ] Simple test circuits
 *
 * Phase 2: Verilog Frontend
 *   [ ] Parse Verilog subset
 *   [ ] Technology mapping to LUT4/LUT6
 *   [ ] Generate netlist
 *
 * Phase 3: Optimization
 *   [ ] Levelization for combinational depth
 *   [ ] Event-driven evaluation
 *   [ ] Memory coalescing for wire state
 *
 * Phase 4: Integration
 *   [ ] VCD waveform output
 *   [ ] Cocotb testbench integration
 *   [ ] Near-memory tiling for huge designs
 *
 * Phase 5: Advanced Features
 *   [ ] Block RAM emulation
 *   [ ] DSP block emulation
 *   [ ] Multiple clock domains
 *   [ ] Timing annotation
 *
 * ═══════════════════════════════════════════════════════════════════════════
 * THE DEEPER MEANING
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * FPGAs are "soft hardware" - circuits defined by configuration.
 * GPUs are "soft parallel processors" - threads defined by code.
 * CPUs are "soft sequential processors" - state machines defined by code.
 *
 * With:
 *   - pseudoscopic (GPU as storage)
 *   - near-memory (GPU + CPU unified access)
 *   - gpuCPU (x86 on GPU)
 *   - gpuFPGA (FPGA on GPU)
 *
 * We've shown that the boundaries are ARBITRARY.
 *
 * Any sufficiently flexible substrate can emulate any computation model.
 * The only differences are:
 *   - Efficiency (how many joules per operation)
 *   - Latency (how fast can we respond)
 *   - Parallelism (how much can we do at once)
 *
 * A GPU with enough memory and the right software is:
 *   - A storage device (pseudoscopic)
 *   - A CPU (gpuCPU)
 *   - An FPGA (gpuFPGA)
 *   - A neural network accelerator (native)
 *
 * It's all just math on bits in memory.
 *
 * 🍪 Cookie Monster's Unified Theory of Computing:
 *    "Me realize: there is no hardware, only software running on physics."
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#ifndef _GPUFPGA_H_
#define _GPUFPGA_H_

#include <stdint.h>
#include <stdbool.h>
#include "nearmem.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ════════════════════════════════════════════════════════════════════════════
 * DATA STRUCTURES
 * ════════════════════════════════════════════════════════════════════════════
 */

/* Maximum configuration */
#define GPUFPGA_MAX_INPUTS     6       /* LUT6 maximum */
#define GPUFPGA_MAX_LUT_SIZE   64      /* 2^6 entries */

/* Wire ID type (supports up to 64K wires per block) */
typedef uint16_t wire_id_t;
#define WIRE_INVALID   0xFFFF

/*
 * LUT descriptor
 * Packed for efficient GPU memory access
 */
typedef struct __attribute__((packed)) {
    uint64_t    lut_mask;                       /* Truth table bits */
    wire_id_t   inputs[GPUFPGA_MAX_INPUTS];     /* Input wire IDs */
    wire_id_t   output;                         /* Output wire ID */
    uint8_t     num_inputs;                     /* Actual input count (2-6) */
    uint8_t     level;                          /* Combinational level (for ordering) */
} gpufpga_lut_t;

/*
 * Flip-flop descriptor
 */
typedef struct __attribute__((packed)) {
    wire_id_t   d_input;        /* D input wire */
    wire_id_t   q_output;       /* Q output wire */
    wire_id_t   clock;          /* Clock wire (for multi-clock) */
    wire_id_t   enable;         /* Enable wire (WIRE_INVALID if none) */
    wire_id_t   reset;          /* Async reset wire (WIRE_INVALID if none) */
    wire_id_t   set;            /* Async set wire (WIRE_INVALID if none) */
    uint8_t     clock_domain;   /* Clock domain ID */
    uint8_t     init_value;     /* Initial value (0 or 1) */
} gpufpga_ff_t;

/*
 * I/O port descriptor
 */
typedef struct __attribute__((packed)) {
    wire_id_t   wire;           /* Internal wire ID */
    uint16_t    bit_index;      /* Bit position in port vector */
    uint8_t     is_input;       /* 1=input, 0=output */
    uint8_t     port_id;        /* Port number */
} gpufpga_io_t;

/*
 * Block RAM descriptor (for memory inference)
 */
typedef struct __attribute__((packed)) {
    uint32_t    data_offset;    /* Offset in RAM data array */
    uint16_t    width;          /* Data width in bits */
    uint16_t    depth;          /* Number of entries */
    wire_id_t   addr_wires[16]; /* Address wire IDs */
    wire_id_t   data_in[64];    /* Data input wire IDs */
    wire_id_t   data_out[64];   /* Data output wire IDs */
    wire_id_t   write_en;       /* Write enable wire */
    wire_id_t   clock;          /* Clock wire */
    uint8_t     addr_bits;      /* Address width */
    uint8_t     flags;          /* Read-first, write-first, etc. */
} gpufpga_bram_t;

/*
 * Circuit header
 */
typedef struct {
    uint32_t    magic;          /* 'GFPG' */
    uint32_t    version;        /* Format version */
    uint32_t    num_luts;       /* Number of LUTs */
    uint32_t    num_ffs;        /* Number of flip-flops */
    uint32_t    num_wires;      /* Total wire count */
    uint32_t    num_ios;        /* Number of I/O ports */
    uint32_t    num_brams;      /* Number of block RAMs */
    uint32_t    num_levels;     /* Combinational depth */
    uint32_t    clock_domains;  /* Number of clock domains */
    uint32_t    flags;          /* Circuit properties */
} gpufpga_header_t;

#define GPUFPGA_MAGIC   0x47465047  /* 'GFPG' */
#define GPUFPGA_VERSION 1

/*
 * Simulation context
 */
typedef struct {
    /* Circuit description (in VRAM) */
    gpufpga_header_t *header;
    gpufpga_lut_t    *luts;
    gpufpga_ff_t     *ffs;
    gpufpga_io_t     *ios;
    gpufpga_bram_t   *brams;
    
    /* Level boundaries (for levelized evaluation) */
    uint32_t         *level_starts;
    
    /* Wire state (in VRAM) */
    uint8_t          *wire_state;       /* Current wire values */
    uint8_t          *wire_next;        /* Next wire values (double buffer) */
    
    /* FF state (in VRAM) */
    uint8_t          *ff_state;         /* Current FF values */
    
    /* Block RAM data (in VRAM) */
    uint8_t          *bram_data;
    
    /* Waveform capture (in VRAM) */
    uint8_t          *waveform;         /* [cycle][wire] */
    uint32_t         waveform_cycles;   /* Cycles captured */
    uint32_t         waveform_stride;   /* Bytes per cycle */
    
    /* Near-memory handles */
    nearmem_ctx_t    *nm_ctx;
    nearmem_region_t circuit_region;
    nearmem_region_t state_region;
    nearmem_region_t waveform_region;
    
    /* Statistics */
    uint64_t         cycles_simulated;
    double           total_time_ms;
    
    /* CUDA resources */
    void             *cuda_stream;
    bool             use_gpu;
} gpufpga_ctx_t;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * BRANCHLESS PRIMITIVES
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * Branchless LUT evaluation
 * No branches, no conditionals - just bit extraction
 */
static inline uint8_t gpufpga_eval_lut(uint64_t lut_mask, uint8_t inputs) {
    return (lut_mask >> inputs) & 1;
}

/*
 * Branchless 2:1 multiplexer
 */
static inline uint8_t gpufpga_mux2(uint8_t a, uint8_t b, uint8_t sel) {
    /* sel must be 0 or 1 */
    return (a & (sel ^ 1)) | (b & sel);
}

/*
 * Branchless flip-flop update
 */
static inline uint8_t gpufpga_ff_update(
    uint8_t q,          /* Current value */
    uint8_t d,          /* D input */
    uint8_t clock_edge, /* 1 if rising edge */
    uint8_t enable,     /* 1 if enabled (or no enable) */
    uint8_t reset,      /* 1 if reset active */
    uint8_t set)        /* 1 if set active */
{
    /* Priority: reset > set > enable > hold */
    uint8_t not_reset = (reset ^ 1) & 1;
    uint8_t not_set = (set ^ 1) & 1;
    uint8_t update = clock_edge & enable & not_reset & not_set;
    
    /* Branchless selection */
    uint8_t result = 0;
    result |= (0 & reset);                      /* Reset wins */
    result |= (1 & set & not_reset);            /* Set if no reset */
    result |= (d & update & not_set);           /* Update if enabled */
    result |= (q & (update ^ 1) & not_reset & not_set);  /* Hold otherwise */
    
    return result & 1;
}

/*
 * Pack multiple 1-bit wires into a multi-bit value
 */
static inline uint64_t gpufpga_pack_wires(
    const uint8_t *wire_state, 
    const wire_id_t *wire_ids, 
    int count)
{
    uint64_t result = 0;
    for (int i = 0; i < count; i++) {
        result |= ((uint64_t)(wire_state[wire_ids[i]] & 1)) << i;
    }
    return result;
}

/*
 * Unpack multi-bit value to individual wire states
 */
static inline void gpufpga_unpack_wires(
    uint8_t *wire_state,
    const wire_id_t *wire_ids,
    uint64_t value,
    int count)
{
    for (int i = 0; i < count; i++) {
        wire_state[wire_ids[i]] = (value >> i) & 1;
    }
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * API FUNCTIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * gpufpga_init - Initialize FPGA simulator
 */
int gpufpga_init(gpufpga_ctx_t *ctx, nearmem_ctx_t *nm_ctx);

/*
 * gpufpga_shutdown - Clean up simulator
 */
void gpufpga_shutdown(gpufpga_ctx_t *ctx);

/*
 * gpufpga_load - Load circuit from netlist file
 */
int gpufpga_load(gpufpga_ctx_t *ctx, const char *filename);

/*
 * gpufpga_load_mem - Load circuit from memory
 */
int gpufpga_load_mem(gpufpga_ctx_t *ctx, const void *data, size_t size);

/*
 * gpufpga_reset - Reset all state to initial values
 */
void gpufpga_reset(gpufpga_ctx_t *ctx);

/*
 * gpufpga_step - Simulate one clock cycle
 */
int gpufpga_step(gpufpga_ctx_t *ctx);

/*
 * gpufpga_run - Simulate multiple cycles
 */
uint64_t gpufpga_run(gpufpga_ctx_t *ctx, uint64_t cycles);

/*
 * gpufpga_set_input - Set input port value
 */
void gpufpga_set_input(gpufpga_ctx_t *ctx, int port_id, uint64_t value);

/*
 * gpufpga_get_output - Get output port value
 */
uint64_t gpufpga_get_output(gpufpga_ctx_t *ctx, int port_id);

/*
 * gpufpga_get_wire - Get single wire value
 */
uint8_t gpufpga_get_wire(gpufpga_ctx_t *ctx, wire_id_t wire);

/*
 * gpufpga_set_wire - Set single wire value (for testbench)
 */
void gpufpga_set_wire(gpufpga_ctx_t *ctx, wire_id_t wire, uint8_t value);

/*
 * gpufpga_enable_waveform - Enable waveform capture
 */
int gpufpga_enable_waveform(gpufpga_ctx_t *ctx, uint32_t max_cycles);

/*
 * gpufpga_save_vcd - Save waveform as VCD file
 */
int gpufpga_save_vcd(gpufpga_ctx_t *ctx, const char *filename);

/*
 * gpufpga_print_stats - Print simulation statistics
 */
void gpufpga_print_stats(gpufpga_ctx_t *ctx);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * CIRCUIT BUILDING API (for programmatic netlist creation)
 * ════════════════════════════════════════════════════════════════════════════
 */

/*
 * Builder context for creating circuits programmatically
 */
typedef struct {
    gpufpga_lut_t   *luts;
    gpufpga_ff_t    *ffs;
    gpufpga_io_t    *ios;
    uint32_t        num_luts, max_luts;
    uint32_t        num_ffs, max_ffs;
    uint32_t        num_ios, max_ios;
    uint32_t        num_wires;
} gpufpga_builder_t;

/*
 * gpufpga_builder_init - Start building a circuit
 */
int gpufpga_builder_init(gpufpga_builder_t *b, 
                          uint32_t max_luts, 
                          uint32_t max_ffs);

/*
 * gpufpga_builder_wire - Allocate a new wire
 */
wire_id_t gpufpga_builder_wire(gpufpga_builder_t *b);

/*
 * gpufpga_builder_input - Add input port
 */
wire_id_t gpufpga_builder_input(gpufpga_builder_t *b, int port_id, int bit);

/*
 * gpufpga_builder_output - Add output port
 */
void gpufpga_builder_output(gpufpga_builder_t *b, int port_id, int bit, wire_id_t wire);

/*
 * gpufpga_builder_lut2 - Add 2-input LUT
 */
wire_id_t gpufpga_builder_lut2(gpufpga_builder_t *b,
                                wire_id_t a, wire_id_t b_in,
                                uint8_t truth_table);

/*
 * gpufpga_builder_lut3 - Add 3-input LUT
 */
wire_id_t gpufpga_builder_lut3(gpufpga_builder_t *b,
                                wire_id_t a, wire_id_t b_in, wire_id_t c,
                                uint8_t truth_table);

/*
 * gpufpga_builder_lut4 - Add 4-input LUT
 */
wire_id_t gpufpga_builder_lut4(gpufpga_builder_t *b,
                                wire_id_t a, wire_id_t b_in, 
                                wire_id_t c, wire_id_t d,
                                uint16_t truth_table);

/*
 * gpufpga_builder_ff - Add flip-flop
 */
wire_id_t gpufpga_builder_ff(gpufpga_builder_t *b,
                              wire_id_t d, wire_id_t clock,
                              wire_id_t enable, wire_id_t reset);

/*
 * Standard gate convenience functions (implemented as LUTs)
 */
wire_id_t gpufpga_builder_and2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in);
wire_id_t gpufpga_builder_or2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in);
wire_id_t gpufpga_builder_xor2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in);
wire_id_t gpufpga_builder_not(gpufpga_builder_t *b, wire_id_t a);
wire_id_t gpufpga_builder_nand2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in);
wire_id_t gpufpga_builder_nor2(gpufpga_builder_t *b, wire_id_t a, wire_id_t b_in);
wire_id_t gpufpga_builder_mux2(gpufpga_builder_t *b, 
                                wire_id_t a, wire_id_t b_in, wire_id_t sel);

/*
 * Multi-bit operations
 */
void gpufpga_builder_adder(gpufpga_builder_t *b,
                            wire_id_t *a, wire_id_t *b_in,
                            wire_id_t *sum, wire_id_t *cout,
                            int bits);

void gpufpga_builder_counter(gpufpga_builder_t *b,
                              wire_id_t clock, wire_id_t reset,
                              wire_id_t *count_out, int bits);

/*
 * gpufpga_builder_finish - Finalize and load into simulator
 */
int gpufpga_builder_finish(gpufpga_builder_t *b, gpufpga_ctx_t *ctx);

/*
 * gpufpga_builder_free - Free builder resources
 */
void gpufpga_builder_free(gpufpga_builder_t *b);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * THERMAL MANAGEMENT SYSTEM
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Accurate chip simulation requires thermal modeling. Every switching event
 * generates heat, and accumulated heat affects performance (thermal throttling)
 * and reliability. This system models:
 *
 *   1. Dynamic Power: P_dyn = α × C × V² × f × N
 *      where α = activity factor (switching rate)
 *      C = capacitance, V = voltage, f = frequency, N = transistor count
 *
 *   2. Static Power: P_static = V × I_leak (temperature dependent)
 *
 *   3. Heat Transfer: Newton's law of cooling
 *      dT/dt = (P_total - h×A×(T - T_ambient)) / (m×c)
 *
 *   4. Thermal Zones: Different regions can have different temperatures
 */

/* Thermal parameters for realistic simulation */
typedef struct {
    float ambient_temp_c;       /* Ambient temperature (°C) */
    float thermal_resistance;   /* Thermal resistance (°C/W) */
    float thermal_capacitance;  /* Thermal capacitance (J/°C) */
    float max_junction_temp;    /* Maximum junction temperature (°C) */
    float throttle_temp;        /* Temperature at which throttling begins */
    float shutdown_temp;        /* Emergency shutdown temperature */
    
    /* Power model parameters */
    float voltage;              /* Supply voltage (V) */
    float base_frequency_mhz;   /* Base clock frequency (MHz) */
    float lut_capacitance_pf;   /* Capacitance per LUT (pF) */
    float ff_capacitance_pf;    /* Capacitance per FF (pF) */
    float static_power_mw;      /* Static power (mW) */
    
    /* Cooling model */
    float heatsink_area_cm2;    /* Heatsink surface area */
    float convection_coeff;     /* Heat transfer coefficient (W/m²·K) */
} gpufpga_thermal_params_t;

/* Default thermal parameters for typical FPGA */
#define GPUFPGA_THERMAL_DEFAULTS { \
    .ambient_temp_c = 25.0f, \
    .thermal_resistance = 1.5f, \
    .thermal_capacitance = 0.8f, \
    .max_junction_temp = 125.0f, \
    .throttle_temp = 95.0f, \
    .shutdown_temp = 110.0f, \
    .voltage = 0.9f, \
    .base_frequency_mhz = 100.0f, \
    .lut_capacitance_pf = 0.5f, \
    .ff_capacitance_pf = 0.2f, \
    .static_power_mw = 500.0f, \
    .heatsink_area_cm2 = 100.0f, \
    .convection_coeff = 25.0f \
}

/* Thermal zone for spatial temperature modeling */
typedef struct {
    uint32_t lut_start, lut_end;    /* LUTs in this zone */
    uint32_t ff_start, ff_end;      /* FFs in this zone */
    float temperature_c;             /* Current temperature */
    float power_mw;                  /* Power dissipation */
    uint64_t switch_count;           /* Switching events this cycle */
    float activity_factor;           /* Rolling average activity */
} gpufpga_thermal_zone_t;

/* Thermal simulation state */
typedef struct {
    gpufpga_thermal_params_t params;
    gpufpga_thermal_zone_t *zones;
    uint32_t num_zones;
    
    /* Per-element activity tracking */
    uint8_t *lut_prev_output;       /* Previous LUT outputs for activity */
    uint8_t *ff_prev_state;         /* Previous FF states for activity */
    
    /* Global thermal state */
    float junction_temp_c;          /* Overall junction temperature */
    float total_power_mw;           /* Total power consumption */
    float throttle_factor;          /* 1.0 = full speed, <1.0 = throttled */
    bool thermal_shutdown;          /* Emergency shutdown triggered */
    
    /* Statistics */
    uint64_t total_switches;        /* Total switching events */
    float peak_temperature_c;       /* Maximum observed temperature */
    float avg_power_mw;             /* Average power consumption */
    uint64_t throttle_cycles;       /* Cycles spent throttled */
} gpufpga_thermal_state_t;

/* Thermal management API */
int gpufpga_thermal_init(gpufpga_ctx_t *ctx, const gpufpga_thermal_params_t *params);
void gpufpga_thermal_shutdown(gpufpga_ctx_t *ctx);
void gpufpga_thermal_step(gpufpga_ctx_t *ctx);
float gpufpga_thermal_get_temperature(gpufpga_ctx_t *ctx);
float gpufpga_thermal_get_power(gpufpga_ctx_t *ctx);
bool gpufpga_thermal_is_throttled(gpufpga_ctx_t *ctx);
void gpufpga_thermal_print_stats(gpufpga_ctx_t *ctx);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * VERILOG/VHDL HDL PARSER
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Load hardware designs directly from Verilog or VHDL source files.
 * Supports a synthesizable subset suitable for FPGA simulation:
 *
 * Supported Verilog constructs:
 *   - module/endmodule definitions
 *   - input/output/wire/reg declarations
 *   - assign statements (combinational logic)
 *   - always @(posedge clk) blocks (sequential logic)
 *   - Basic operators: &, |, ^, ~, +, -, *, ?, :
 *   - if/else, case statements
 *   - Parameterized modules
 *
 * Supported VHDL constructs:
 *   - entity/architecture definitions
 *   - signal declarations
 *   - concurrent signal assignments
 *   - process blocks with sensitivity lists
 *   - Basic operators and functions
 *
 * The parser performs:
 *   1. Lexing and parsing to AST
 *   2. Elaboration (parameter substitution, generate blocks)
 *   3. Technology mapping to LUT/FF primitives
 *   4. Netlist optimization (constant propagation, dead code removal)
 *   5. Levelization for efficient simulation
 */

/* HDL language type */
typedef enum {
    GPUFPGA_HDL_VERILOG,        /* Verilog-2001 subset */
    GPUFPGA_HDL_SYSTEMVERILOG,  /* SystemVerilog subset */
    GPUFPGA_HDL_VHDL,           /* VHDL-93 subset */
    GPUFPGA_HDL_AUTO            /* Auto-detect from file extension */
} gpufpga_hdl_lang_t;

/* HDL compilation options */
typedef struct {
    gpufpga_hdl_lang_t language;
    const char *top_module;         /* Top-level module name (NULL = auto) */
    const char **include_paths;     /* Include search paths */
    uint32_t num_include_paths;
    const char **defines;           /* Preprocessor defines (NAME=VALUE) */
    uint32_t num_defines;
    
    /* Optimization options */
    bool optimize_constants;        /* Propagate constant values */
    bool optimize_dead_code;        /* Remove unused logic */
    bool optimize_retiming;         /* Move FFs for better timing */
    uint8_t lut_size;               /* Target LUT size (4 or 6) */
    
    /* Debug options */
    bool keep_hierarchy;            /* Preserve module hierarchy */
    bool generate_debug_info;       /* Include source location mapping */
    int verbosity;                  /* 0=quiet, 1=normal, 2=verbose */
} gpufpga_hdl_options_t;

/* Default HDL compilation options */
#define GPUFPGA_HDL_OPTIONS_DEFAULT { \
    .language = GPUFPGA_HDL_AUTO, \
    .top_module = NULL, \
    .include_paths = NULL, \
    .num_include_paths = 0, \
    .defines = NULL, \
    .num_defines = 0, \
    .optimize_constants = true, \
    .optimize_dead_code = true, \
    .optimize_retiming = false, \
    .lut_size = 4, \
    .keep_hierarchy = false, \
    .generate_debug_info = true, \
    .verbosity = 1 \
}

/* HDL compilation result */
typedef struct {
    bool success;
    uint32_t num_errors;
    uint32_t num_warnings;
    char **error_messages;
    char **warning_messages;
    
    /* Statistics */
    uint32_t input_lines;           /* Source lines processed */
    uint32_t num_modules;           /* Modules in design */
    uint32_t num_instances;         /* Module instances */
    double compile_time_ms;         /* Compilation time */
    
    /* Mapping statistics */
    uint32_t luts_generated;
    uint32_t ffs_generated;
    uint32_t wires_generated;
    uint8_t max_comb_depth;         /* Maximum combinational depth */
} gpufpga_hdl_result_t;

/* Port mapping for testbench integration */
typedef struct {
    const char *name;       /* Port name from HDL */
    int port_id;            /* Assigned port ID */
    int width;              /* Bit width */
    bool is_input;          /* Direction */
} gpufpga_port_map_t;

/*
 * gpufpga_load_verilog - Load circuit from Verilog source
 *
 * @ctx: Simulator context
 * @filename: Verilog source file path
 * @options: Compilation options (NULL for defaults)
 * @result: Compilation result (optional, can be NULL)
 *
 * Returns: 0 on success, negative error code on failure
 */
int gpufpga_load_verilog(gpufpga_ctx_t *ctx,
                          const char *filename,
                          const gpufpga_hdl_options_t *options,
                          gpufpga_hdl_result_t *result);

/*
 * gpufpga_load_vhdl - Load circuit from VHDL source
 */
int gpufpga_load_vhdl(gpufpga_ctx_t *ctx,
                       const char *filename,
                       const gpufpga_hdl_options_t *options,
                       gpufpga_hdl_result_t *result);

/*
 * gpufpga_load_hdl_multi - Load circuit from multiple HDL files
 *
 * Supports mixed Verilog/VHDL designs with proper module resolution.
 */
int gpufpga_load_hdl_multi(gpufpga_ctx_t *ctx,
                            const char **filenames,
                            uint32_t num_files,
                            const gpufpga_hdl_options_t *options,
                            gpufpga_hdl_result_t *result);

/*
 * gpufpga_get_port_map - Get I/O port mapping from loaded HDL
 *
 * After loading HDL, use this to discover the available ports
 * for testbench integration.
 */
int gpufpga_get_port_map(gpufpga_ctx_t *ctx,
                          gpufpga_port_map_t **ports,
                          uint32_t *num_ports);

/*
 * gpufpga_find_port - Find port by name
 */
int gpufpga_find_port(gpufpga_ctx_t *ctx, const char *name);

/*
 * gpufpga_free_result - Free compilation result resources
 */
void gpufpga_free_result(gpufpga_hdl_result_t *result);

/*
 * gpufpga_save_vcd - Save waveform as Value Change Dump
 *
 * Standard VCD format compatible with GTKWave and other viewers.
 * When compiled from HDL, uses original signal names.
 */
int gpufpga_save_vcd(gpufpga_ctx_t *ctx, const char *filename);

/*
 * ════════════════════════════════════════════════════════════════════════════
 * TIMING ANALYSIS
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Static timing analysis for the emulated design.
 */

typedef struct {
    float setup_time_ns;        /* Setup time requirement */
    float hold_time_ns;         /* Hold time requirement */
    float clock_to_q_ns;        /* Clock-to-Q delay */
    float lut_delay_ns;         /* LUT propagation delay */
    float routing_delay_ns;     /* Average routing delay */
    float max_frequency_mhz;    /* Maximum achievable frequency */
    
    /* Critical path information */
    wire_id_t *critical_path;   /* Wire IDs on critical path */
    uint32_t critical_path_len;
    float critical_path_ns;     /* Critical path delay */
} gpufpga_timing_t;

int gpufpga_analyze_timing(gpufpga_ctx_t *ctx, gpufpga_timing_t *timing);
void gpufpga_free_timing(gpufpga_timing_t *timing);

#ifdef __cplusplus
}
#endif

#endif /* _GPUFPGA_H_ */

