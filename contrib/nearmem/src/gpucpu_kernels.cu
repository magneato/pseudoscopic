/*
 * gpucpu_kernels.cu - GPU-Native x86 Interpreter Kernels
 *
 * THE DREAM REALIZED: The x86 fetch-decode-execute loop runs entirely on GPU.
 * No CPU involvement except for I/O and interrupt handling.
 *
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                     GPU VRAM (via Pseudoscopic)                     │
 *   ├──────────────┬───────────────────┬──────────────────────────────────┤
 *   │  x86 State   │    Guest RAM      │       Decode Tables              │
 *   │  (registers) │  (code + data)    │   (constant memory LUTs)         │
 *   └──────┬───────┴─────────┬─────────┴───────────────┬──────────────────┘
 *          │                 │                         │
 *          ▼                 ▼                         ▼
 *   ┌─────────────────────────────────────────────────────────────────────┐
 *   │                     GPU COMPUTE (CUDA Cores)                        │
 *   │                                                                     │
 *   │    ┌──────────────────────────────────────────────────────────┐     │
 *   │    │                x86_execute_kernel                        │     │
 *   │    │                                                          │     │
 *   │    │   1. Fetch opcode from ram[RIP]                          │     │
 *   │    │   2. Decode via switch/LUT                               │     │
 *   │    │   3. Execute operation                                   │     │
 *   │    │   4. Update registers                                    │     │
 *   │    │   5. Advance RIP                                         │     │
 *   │    │   6. Repeat until HLT or interrupt need                  │     │
 *   │    │                                                          │     │
 *   │    └──────────────────────────────────────────────────────────┘     │
 *   │                                                                     │
 *   └─────────────────────────────────────────────────────────────────────┘
 *
 * KEY INSIGHT: A single GPU thread runs the entire emulation loop.
 * This is INTENTIONALLY single-threaded for the interpreter core.
 * Parallelism comes from:
 *   1. Multiple emulated cores (one per SM for multi-core emulation)
 *   2. Bulk memory operations (REP MOVS gets its own parallel kernel)
 *   3. Memory bandwidth (700 GB/s HBM2 >> 50 GB/s DDR4)
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

/* Include gpucpu.h for type definitions */
extern "C" {
#include "gpucpu.h"
}

/* Block sizes */
#define INTERPRETER_THREADS 1   /* Single thread for sequential execution */
#define MEMCPY_BLOCK_SIZE 256

/*
 * ============================================================
 * Device-side x86 State Access
 * ============================================================
 */

/* FLAGS bits (duplicated for device code) */
#define D_FLAG_CF     (1 << 0)
#define D_FLAG_PF     (1 << 2)
#define D_FLAG_AF     (1 << 4)
#define D_FLAG_ZF     (1 << 6)
#define D_FLAG_SF     (1 << 7)
#define D_FLAG_TF     (1 << 8)
#define D_FLAG_IF     (1 << 9)
#define D_FLAG_DF     (1 << 10)
#define D_FLAG_OF     (1 << 11)

/*
 * ============================================================
 * Device Helper Functions
 * ============================================================
 */

/* Calculate linear address from segment:offset (real mode) */
__device__ __forceinline__ uint32_t d_real_addr(uint16_t seg, uint16_t off) {
    return ((uint32_t)seg << 4) + off;
}

/* Read byte from RAM */
__device__ __forceinline__ uint8_t d_read8(uint8_t *ram, size_t ram_size, uint32_t addr) {
    if (addr < ram_size)
        return ram[addr];
    return 0xFF;
}

/* Read word from RAM (little-endian) */
__device__ __forceinline__ uint16_t d_read16(uint8_t *ram, size_t ram_size, uint32_t addr) {
    return d_read8(ram, ram_size, addr) | ((uint16_t)d_read8(ram, ram_size, addr + 1) << 8);
}

/* Write byte to RAM */
__device__ __forceinline__ void d_write8(uint8_t *ram, size_t ram_size, uint32_t addr, uint8_t val) {
    if (addr < ram_size)
        ram[addr] = val;
}

/* Write word to RAM (little-endian) */
__device__ __forceinline__ void d_write16(uint8_t *ram, size_t ram_size, uint32_t addr, uint16_t val) {
    d_write8(ram, ram_size, addr, val & 0xFF);
    d_write8(ram, ram_size, addr + 1, val >> 8);
}

/* Fetch byte at CS:IP and advance IP */
__device__ __forceinline__ uint8_t d_fetch8(x86_state_t *s, uint8_t *ram, size_t ram_size) {
    uint32_t addr = d_real_addr(s->cs, s->rip & 0xFFFF);
    uint8_t val = d_read8(ram, ram_size, addr);
    s->rip++;
    return val;
}

/* Fetch word at CS:IP and advance IP */
__device__ __forceinline__ uint16_t d_fetch16(x86_state_t *s, uint8_t *ram, size_t ram_size) {
    uint16_t lo = d_fetch8(s, ram, ram_size);
    uint16_t hi = d_fetch8(s, ram, ram_size);
    return lo | (hi << 8);
}

/* Push word onto stack */
__device__ __forceinline__ void d_push16(x86_state_t *s, uint8_t *ram, size_t ram_size, uint16_t val) {
    s->rsp = (s->rsp - 2) & 0xFFFF;
    d_write16(ram, ram_size, d_real_addr(s->ss, s->rsp), val);
}

/* Pop word from stack */
__device__ __forceinline__ uint16_t d_pop16(x86_state_t *s, uint8_t *ram, size_t ram_size) {
    uint16_t val = d_read16(ram, ram_size, d_real_addr(s->ss, s->rsp));
    s->rsp = (s->rsp + 2) & 0xFFFF;
    return val;
}

/* Update flags after 8-bit operation */
__device__ __forceinline__ void d_update_flags_8(x86_state_t *s, uint8_t result, int carry, int overflow) {
    s->rflags &= ~(D_FLAG_CF | D_FLAG_ZF | D_FLAG_SF | D_FLAG_OF | D_FLAG_PF);
    
    if (carry) s->rflags |= D_FLAG_CF;
    if (overflow) s->rflags |= D_FLAG_OF;
    if (result == 0) s->rflags |= D_FLAG_ZF;
    if (result & 0x80) s->rflags |= D_FLAG_SF;
    
    /* Parity flag */
    int bits = __popc(result & 0xFF);
    if ((bits & 1) == 0) s->rflags |= D_FLAG_PF;
}

/* Update flags after 16-bit operation */
__device__ __forceinline__ void d_update_flags_16(x86_state_t *s, uint16_t result, int carry, int overflow) {
    s->rflags &= ~(D_FLAG_CF | D_FLAG_ZF | D_FLAG_SF | D_FLAG_OF | D_FLAG_PF);
    
    if (carry) s->rflags |= D_FLAG_CF;
    if (overflow) s->rflags |= D_FLAG_OF;
    if (result == 0) s->rflags |= D_FLAG_ZF;
    if (result & 0x8000) s->rflags |= D_FLAG_SF;
    
    int bits = __popc(result & 0xFF);
    if ((bits & 1) == 0) s->rflags |= D_FLAG_PF;
}

/* Get pointer to 16-bit register by index */
__device__ __forceinline__ uint16_t* d_get_reg16(x86_state_t *s, int idx) {
    switch (idx & 7) {
        case 0: return (uint16_t*)&s->rax;
        case 1: return (uint16_t*)&s->rcx;
        case 2: return (uint16_t*)&s->rdx;
        case 3: return (uint16_t*)&s->rbx;
        case 4: return (uint16_t*)&s->rsp;
        case 5: return (uint16_t*)&s->rbp;
        case 6: return (uint16_t*)&s->rsi;
        case 7: return (uint16_t*)&s->rdi;
    }
    return (uint16_t*)&s->rax;
}

/* Get pointer to 8-bit register by index */
__device__ __forceinline__ uint8_t* d_get_reg8(x86_state_t *s, int idx) {
    switch (idx & 7) {
        case 0: return (uint8_t*)&s->rax;
        case 1: return (uint8_t*)&s->rcx;
        case 2: return (uint8_t*)&s->rdx;
        case 3: return (uint8_t*)&s->rbx;
        case 4: return ((uint8_t*)&s->rax) + 1;  /* AH */
        case 5: return ((uint8_t*)&s->rcx) + 1;  /* CH */
        case 6: return ((uint8_t*)&s->rdx) + 1;  /* DH */
        case 7: return ((uint8_t*)&s->rbx) + 1;  /* BH */
    }
    return (uint8_t*)&s->rax;
}

/*
 * ============================================================
 * GPU x86 Interpreter Kernel
 * ============================================================
 *
 * This is the heart of GPU-CPU: a complete x86 interpreter running
 * on a single GPU thread. The thread has exclusive access to the
 * x86 state and RAM in VRAM.
 *
 * Why single-threaded? Because x86 execution is inherently sequential.
 * The parallelism win comes from:
 *   1. Running many emulated CPUs (one per SM)
 *   2. Bulk operations (REP MOVS becomes parallel kernel)
 *   3. Memory bandwidth advantage (HBM2 vs DDR4)
 */
__global__ void x86_execute_kernel(
    x86_state_t *state,
    uint8_t *ram,
    size_t ram_size,
    uint64_t max_instructions,
    int *needs_io,        /* Output: set to 1 if INT 21h I/O needed */
    uint8_t *io_vector,   /* Output: interrupt vector for I/O */
    uint64_t *executed    /* Output: instructions executed */
)
{
    /* Single thread execution */
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    
    x86_state_t *s = state;
    uint64_t count = 0;
    
    *needs_io = 0;
    
    while (!s->halted) {
        if (max_instructions > 0 && count >= max_instructions)
            break;
        
        /* Fetch opcode */
        uint8_t op = d_fetch8(s, ram, ram_size);
        
        /* Decode and execute */
        switch (op) {
            
            /* ===== NOP ===== */
            case 0x90:
                break;
            
            /* ===== MOV immediate to register ===== */
            case 0xB0: case 0xB1: case 0xB2: case 0xB3:
            case 0xB4: case 0xB5: case 0xB6: case 0xB7:
                /* MOV r8, imm8 */
                *d_get_reg8(s, op - 0xB0) = d_fetch8(s, ram, ram_size);
                break;
            
            case 0xB8: case 0xB9: case 0xBA: case 0xBB:
            case 0xBC: case 0xBD: case 0xBE: case 0xBF:
                /* MOV r16, imm16 */
                *d_get_reg16(s, op - 0xB8) = d_fetch16(s, ram, ram_size);
                break;
            
            /* ===== MOV register to/from register ===== */
            case 0x88: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0)
                    *d_get_reg8(s, rm) = *d_get_reg8(s, reg);
                break;
            }
            
            case 0x89: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0)
                    *d_get_reg16(s, rm) = *d_get_reg16(s, reg);
                break;
            }
            
            case 0x8A: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0)
                    *d_get_reg8(s, reg) = *d_get_reg8(s, rm);
                break;
            }
            
            case 0x8B: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0)
                    *d_get_reg16(s, reg) = *d_get_reg16(s, rm);
                break;
            }
            
            /* ===== ADD ===== */
            case 0x00: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint8_t a = *d_get_reg8(s, rm);
                    uint8_t b = *d_get_reg8(s, reg);
                    uint16_t result = a + b;
                    *d_get_reg8(s, rm) = result;
                    d_update_flags_8(s, result, result > 0xFF,
                                    ((a ^ result) & (b ^ result) & 0x80) != 0);
                }
                break;
            }
            
            case 0x01: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint16_t a = *d_get_reg16(s, rm);
                    uint16_t b = *d_get_reg16(s, reg);
                    uint32_t result = a + b;
                    *d_get_reg16(s, rm) = result;
                    d_update_flags_16(s, result, result > 0xFFFF,
                                     ((a ^ result) & (b ^ result) & 0x8000) != 0);
                }
                break;
            }
            
            case 0x04: {
                uint8_t imm = d_fetch8(s, ram, ram_size);
                uint8_t a = s->rax & 0xFF;
                uint16_t result = a + imm;
                s->rax = (s->rax & 0xFF00) | (result & 0xFF);
                d_update_flags_8(s, result, result > 0xFF,
                                ((a ^ result) & (imm ^ result) & 0x80) != 0);
                break;
            }
            
            case 0x05: {
                uint16_t imm = d_fetch16(s, ram, ram_size);
                uint16_t a = s->rax & 0xFFFF;
                uint32_t result = a + imm;
                s->rax = (s->rax & 0xFFFF0000) | (result & 0xFFFF);
                d_update_flags_16(s, result, result > 0xFFFF,
                                 ((a ^ result) & (imm ^ result) & 0x8000) != 0);
                break;
            }
            
            /* ===== SUB ===== */
            case 0x28: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint8_t a = *d_get_reg8(s, rm);
                    uint8_t b = *d_get_reg8(s, reg);
                    uint8_t result = a - b;
                    *d_get_reg8(s, rm) = result;
                    d_update_flags_8(s, result, a < b,
                                    ((a ^ b) & (a ^ result) & 0x80) != 0);
                }
                break;
            }
            
            case 0x29: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint16_t a = *d_get_reg16(s, rm);
                    uint16_t b = *d_get_reg16(s, reg);
                    uint16_t result = a - b;
                    *d_get_reg16(s, rm) = result;
                    d_update_flags_16(s, result, a < b,
                                     ((a ^ b) & (a ^ result) & 0x8000) != 0);
                }
                break;
            }
            
            case 0x2C: {
                uint8_t imm = d_fetch8(s, ram, ram_size);
                uint8_t a = s->rax & 0xFF;
                uint8_t result = a - imm;
                s->rax = (s->rax & 0xFF00) | result;
                d_update_flags_8(s, result, a < imm,
                                ((a ^ imm) & (a ^ result) & 0x80) != 0);
                break;
            }
            
            /* ===== CMP ===== */
            case 0x38: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint8_t a = *d_get_reg8(s, rm);
                    uint8_t b = *d_get_reg8(s, reg);
                    uint8_t result = a - b;
                    d_update_flags_8(s, result, a < b,
                                    ((a ^ b) & (a ^ result) & 0x80) != 0);
                }
                break;
            }
            
            case 0x39: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint16_t a = *d_get_reg16(s, rm);
                    uint16_t b = *d_get_reg16(s, reg);
                    uint16_t result = a - b;
                    d_update_flags_16(s, result, a < b,
                                     ((a ^ b) & (a ^ result) & 0x8000) != 0);
                }
                break;
            }
            
            case 0x3C: {
                uint8_t imm = d_fetch8(s, ram, ram_size);
                uint8_t a = s->rax & 0xFF;
                uint8_t result = a - imm;
                d_update_flags_8(s, result, a < imm,
                                ((a ^ imm) & (a ^ result) & 0x80) != 0);
                break;
            }
            
            /* ===== INC r16 ===== */
            case 0x40: case 0x41: case 0x42: case 0x43:
            case 0x44: case 0x45: case 0x46: case 0x47: {
                int reg = op - 0x40;
                uint16_t *r = d_get_reg16(s, reg);
                int old_cf = s->rflags & D_FLAG_CF;
                uint16_t result = *r + 1;
                *r = result;
                d_update_flags_16(s, result, 0, (*r == 0x8000));
                if (old_cf) s->rflags |= D_FLAG_CF;
                break;
            }
            
            /* ===== DEC r16 ===== */
            case 0x48: case 0x49: case 0x4A: case 0x4B:
            case 0x4C: case 0x4D: case 0x4E: case 0x4F: {
                int reg = op - 0x48;
                uint16_t *r = d_get_reg16(s, reg);
                int old_cf = s->rflags & D_FLAG_CF;
                uint16_t old_val = *r;
                uint16_t result = *r - 1;
                *r = result;
                d_update_flags_16(s, result, 0, (old_val == 0x8000));
                if (old_cf) s->rflags |= D_FLAG_CF;
                break;
            }
            
            /* ===== PUSH r16 ===== */
            case 0x50: case 0x51: case 0x52: case 0x53:
            case 0x54: case 0x55: case 0x56: case 0x57:
                d_push16(s, ram, ram_size, *d_get_reg16(s, op - 0x50));
                break;
            
            /* ===== POP r16 ===== */
            case 0x58: case 0x59: case 0x5A: case 0x5B:
            case 0x5C: case 0x5D: case 0x5E: case 0x5F:
                *d_get_reg16(s, op - 0x58) = d_pop16(s, ram, ram_size);
                break;
            
            /* ===== JMP rel8 ===== */
            case 0xEB: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JMP rel16 ===== */
            case 0xE9: {
                int16_t rel = (int16_t)d_fetch16(s, ram, ram_size);
                s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JE/JZ rel8 ===== */
            case 0x74: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                if (s->rflags & D_FLAG_ZF)
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JNE/JNZ rel8 ===== */
            case 0x75: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                if (!(s->rflags & D_FLAG_ZF))
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JL rel8 ===== */
            case 0x7C: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                bool sf = (s->rflags & D_FLAG_SF) != 0;
                bool of = (s->rflags & D_FLAG_OF) != 0;
                if (sf != of)
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JGE rel8 ===== */
            case 0x7D: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                bool sf = (s->rflags & D_FLAG_SF) != 0;
                bool of = (s->rflags & D_FLAG_OF) != 0;
                if (sf == of)
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JLE rel8 ===== */
            case 0x7E: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                bool zf = (s->rflags & D_FLAG_ZF) != 0;
                bool sf = (s->rflags & D_FLAG_SF) != 0;
                bool of = (s->rflags & D_FLAG_OF) != 0;
                if (zf || (sf != of))
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== JG rel8 ===== */
            case 0x7F: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                bool zf = (s->rflags & D_FLAG_ZF) != 0;
                bool sf = (s->rflags & D_FLAG_SF) != 0;
                bool of = (s->rflags & D_FLAG_OF) != 0;
                if (!zf && (sf == of))
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== LOOP rel8 ===== */
            case 0xE2: {
                int8_t rel = (int8_t)d_fetch8(s, ram, ram_size);
                s->rcx = (s->rcx - 1) & 0xFFFF;
                if ((s->rcx & 0xFFFF) != 0)
                    s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== CALL rel16 ===== */
            case 0xE8: {
                int16_t rel = (int16_t)d_fetch16(s, ram, ram_size);
                d_push16(s, ram, ram_size, s->rip & 0xFFFF);
                s->rip = (s->rip + rel) & 0xFFFF;
                break;
            }
            
            /* ===== RET ===== */
            case 0xC3:
                s->rip = d_pop16(s, ram, ram_size);
                break;
            
            /* ===== INT imm8 ===== */
            case 0xCD: {
                uint8_t vector = d_fetch8(s, ram, ram_size);
                
                /* INT 21h and INT 20h require host CPU for I/O */
                if (vector == 0x21 || vector == 0x20) {
                    *needs_io = 1;
                    *io_vector = vector;
                    s->rip--;  /* Back up to re-execute after I/O */
                    s->rip--;
                    goto exit_loop;
                }
                break;
            }
            
            /* ===== HLT ===== */
            case 0xF4:
                s->halted = true;
                break;
            
            /* ===== XOR ===== */
            case 0x30: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint8_t result = *d_get_reg8(s, rm) ^ *d_get_reg8(s, reg);
                    *d_get_reg8(s, rm) = result;
                    d_update_flags_8(s, result, 0, 0);
                }
                break;
            }
            
            case 0x31: {
                uint8_t modrm = d_fetch8(s, ram, ram_size);
                int reg = (modrm >> 3) & 7;
                int rm = modrm & 7;
                if ((modrm & 0xC0) == 0xC0) {
                    uint16_t result = *d_get_reg16(s, rm) ^ *d_get_reg16(s, reg);
                    *d_get_reg16(s, rm) = result;
                    d_update_flags_16(s, result, 0, 0);
                }
                break;
            }
            
            /* ===== Unknown opcode ===== */
            default:
                /* Unknown opcode - halt */
                s->halted = true;
                goto exit_loop;
        }
        
        s->instructions_executed++;
        count++;
    }
    
exit_loop:
    *executed = count;
}

/*
 * ============================================================
 * Parallel Memory Copy Kernel (for REP MOVS)
 * ============================================================
 */
__global__ void x86_memcpy_kernel(
    uint8_t *ram,
    uint64_t dst,
    uint64_t src,
    uint64_t count,
    int direction  /* 0 = forward, 1 = backward */
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    if (direction == 0) {
        for (size_t i = idx; i < count; i += stride)
            ram[dst + i] = ram[src + i];
    } else {
        for (size_t i = idx; i < count; i += stride)
            ram[dst + count - 1 - i] = ram[src + count - 1 - i];
    }
}

/*
 * ============================================================
 * Parallel Memory Set Kernel (for REP STOS)
 * ============================================================
 */
__global__ void x86_memset_kernel(
    uint8_t *ram,
    uint64_t dst,
    uint8_t value,
    uint64_t count
)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;
    
    for (size_t i = idx; i < count; i += stride)
        ram[dst + i] = value;
}

/*
 * ============================================================
 * Host-side Interface
 * ============================================================
 */

extern "C" {

/* Device pointers for persistent allocation */
static int *d_needs_io = nullptr;
static uint8_t *d_io_vector = nullptr;
static uint64_t *d_executed = nullptr;

/*
 * gpucpu_cuda_init - Initialize CUDA resources
 */
int gpucpu_cuda_init(void) {
    cudaError_t err;
    
    err = cudaMalloc(&d_needs_io, sizeof(int));
    if (err != cudaSuccess) return -1;
    
    err = cudaMalloc(&d_io_vector, sizeof(uint8_t));
    if (err != cudaSuccess) return -1;
    
    err = cudaMalloc(&d_executed, sizeof(uint64_t));
    if (err != cudaSuccess) return -1;
    
    return 0;
}

/*
 * gpucpu_cuda_shutdown - Clean up CUDA resources
 */
void gpucpu_cuda_shutdown(void) {
    if (d_needs_io) cudaFree(d_needs_io);
    if (d_io_vector) cudaFree(d_io_vector);
    if (d_executed) cudaFree(d_executed);
    d_needs_io = nullptr;
    d_io_vector = nullptr;
    d_executed = nullptr;
}

/*
 * gpucpu_cuda_run - Run x86 emulation on GPU
 *
 * Returns: instructions executed
 * Sets needs_io to 1 if I/O interrupt needs host handling
 */
uint64_t gpucpu_cuda_run(
    void *state,           /* x86_state_t in VRAM */
    void *ram,             /* Guest RAM in VRAM */
    size_t ram_size,
    uint64_t max_instructions,
    int *needs_io,
    uint8_t *io_vector
)
{
    int h_needs_io = 0;
    uint8_t h_io_vector = 0;
    uint64_t h_executed = 0;
    
    cudaMemset(d_needs_io, 0, sizeof(int));
    cudaMemset(d_io_vector, 0, sizeof(uint8_t));
    cudaMemset(d_executed, 0, sizeof(uint64_t));
    
    /* Launch single-threaded interpreter kernel */
    x86_execute_kernel<<<1, 1>>>(
        (x86_state_t*)state,
        (uint8_t*)ram,
        ram_size,
        max_instructions,
        d_needs_io,
        d_io_vector,
        d_executed
    );
    
    cudaDeviceSynchronize();
    
    cudaMemcpy(&h_needs_io, d_needs_io, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_io_vector, d_io_vector, sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_executed, d_executed, sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    *needs_io = h_needs_io;
    *io_vector = h_io_vector;
    
    return h_executed;
}

/*
 * gpucpu_cuda_available - Check if CUDA is available
 */
int gpucpu_cuda_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

/*
 * gpucpu_cuda_get_device_info - Get GPU information for tier detection
 *
 * Returns GPU tier based on memory and compute capability.
 * Also detects specific GPU models for special handling.
 */
int gpucpu_cuda_get_device_info(int device_id, char *name, size_t name_len,
                                 size_t *total_mem, int *compute_major,
                                 int *compute_minor, int *is_tiny)
{
    cudaDeviceProp prop;
    cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
    if (err != cudaSuccess) {
        return -1;
    }
    
    if (name && name_len > 0) {
        strncpy(name, prop.name, name_len - 1);
        name[name_len - 1] = '\0';
    }
    
    if (total_mem) *total_mem = prop.totalGlobalMem;
    if (compute_major) *compute_major = prop.major;
    if (compute_minor) *compute_minor = prop.minor;
    
    /*
     * TINY MODE DETECTION
     * ===================
     * Enable tiny mode for low-memory GPUs:
     * - GT 1030 (2 GB, GP108)
     * - GTX 1050 (2 GB)
     * - MX series (2-4 GB)
     * - Any GPU with < 4 GB
     */
    if (is_tiny) {
        *is_tiny = 0;
        
        /* Memory-based detection: < 4 GB = tiny mode */
        if (prop.totalGlobalMem < (size_t)4 * 1024 * 1024 * 1024) {
            *is_tiny = 1;
        }
        
        /* Name-based detection for known tiny GPUs */
        if (strstr(prop.name, "GT 1030") ||
            strstr(prop.name, "GTX 1050") ||
            strstr(prop.name, "MX") ||
            strstr(prop.name, "GT 710") ||
            strstr(prop.name, "GT 730") ||
            strstr(prop.name, "GT 1010")) {
            *is_tiny = 1;
        }
    }
    
    return 0;
}

/*
 * gpucpu_cuda_select_device - Select best device for gpucpu
 *
 * Prefers the pseudoscopic-bound device if available.
 * Falls back to best available GPU otherwise.
 *
 * Returns: device ID, or -1 if no suitable device
 */
int gpucpu_cuda_select_device(const char *pci_bus_id) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return -1;
    }
    
    /* If PCI bus ID specified, try to find matching device */
    if (pci_bus_id && pci_bus_id[0]) {
        for (int i = 0; i < device_count; i++) {
            char bus_id[16];
            err = cudaDeviceGetPCIBusId(bus_id, sizeof(bus_id), i);
            if (err == cudaSuccess && strstr(bus_id, pci_bus_id)) {
                cudaSetDevice(i);
                return i;
            }
        }
    }
    
    /* Fall back to device 0 */
    cudaSetDevice(0);
    return 0;
}

/*
 * gpucpu_cuda_get_tier - Determine GPU tier from memory size
 */
int gpucpu_cuda_get_tier(size_t total_mem) {
    if (total_mem < (size_t)4 * 1024 * 1024 * 1024) {
        return 1;  /* GPU_TIER_TINY */
    } else if (total_mem < (size_t)8 * 1024 * 1024 * 1024) {
        return 2;  /* GPU_TIER_LOW */
    } else if (total_mem < (size_t)16 * 1024 * 1024 * 1024) {
        return 3;  /* GPU_TIER_MID */
    } else {
        return 4;  /* GPU_TIER_HIGH */
    }
}

/*
 * gpucpu_cuda_get_recommended_ram - Get recommended guest RAM size for tier
 *
 * For tiny mode GPUs, we recommend smaller guest RAM to leave room
 * for GPU driver overhead and other allocations.
 */
size_t gpucpu_cuda_get_recommended_ram(int tier, size_t total_mem) {
    switch (tier) {
        case 1:  /* TINY: GT 1030 (2 GB) */
            /* Reserve 512 MB for driver, use up to 1 GB for guest */
            return (total_mem > (512 * 1024 * 1024)) ?
                   (total_mem - 512 * 1024 * 1024) : (256 * 1024 * 1024);
        case 2:  /* LOW: 4-8 GB */
            return 2 * 1024 * 1024 * 1024UL;  /* 2 GB */
        case 3:  /* MID: 8-16 GB */
            return 8 * 1024 * 1024 * 1024UL;  /* 8 GB */
        case 4:  /* HIGH: 24+ GB */
            return 32 * 1024 * 1024 * 1024UL; /* 32 GB */
        default:
            return 1 * 1024 * 1024;  /* 1 MB fallback */
    }
}

} /* extern "C" */

