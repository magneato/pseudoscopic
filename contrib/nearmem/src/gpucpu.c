/*
 * gpucpu.c - GPU-CPU Emulator: x86 on GPU
 *
 * THE DREAM REALIZED: x86 execution runs ENTIRELY on GPU.
 *
 * This implements the x86 interpreter with two execution paths:
 *   1. GPU Path (preferred): The fetch-decode-execute loop runs on GPU
 *   2. CPU Path (fallback): Traditional CPU interpretation
 *
 * The GPU path uses gpucpu_kernels.cu for the interpreter kernel.
 * I/O operations (INT 21h) trap to host CPU, then return to GPU.
 *
 * "The GPU IS the CPU. We've just been using it wrong."
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <sys/time.h>

#include "gpucpu.h"

/*
 * CUDA interface declarations (implemented in gpucpu_kernels.cu)
 */
#ifdef NEARMEM_HAS_CUDA
extern int gpucpu_cuda_init(void);
extern void gpucpu_cuda_shutdown(void);
extern uint64_t gpucpu_cuda_run(void *state, void *ram, size_t ram_size,
                                 uint64_t max_instructions,
                                 int *needs_io, uint8_t *io_vector);
extern int gpucpu_cuda_available(void);
extern int gpucpu_cuda_get_device_info(int device_id, char *name, size_t name_len,
                                        size_t *total_mem, int *compute_major,
                                        int *compute_minor, int *is_tiny);
extern int gpucpu_cuda_select_device(const char *pci_bus_id);
extern int gpucpu_cuda_get_tier(size_t total_mem);
extern size_t gpucpu_cuda_get_recommended_ram(int tier, size_t total_mem);
#endif

/*
 * Timing helper
 */
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/*
 * ============================================================
 * Initialization
 * ============================================================
 */

int gpucpu_init(gpucpu_ctx_t *ctx, nearmem_ctx_t *nm_ctx, size_t ram_size) {
    if (!ctx || !nm_ctx)
        return -1;
    
    memset(ctx, 0, sizeof(*ctx));
    ctx->memory.nm_ctx = nm_ctx;
    ctx->memory.ram_size = ram_size;
    
    /* Allocate state structure in VRAM */
    size_t state_size = sizeof(x86_state_t);
    state_size = (state_size + 4095) & ~4095;  /* Page align */
    
    if (nearmem_alloc(nm_ctx, &ctx->memory.state_region, state_size) != NEARMEM_OK) {
        fprintf(stderr, "gpucpu: Failed to allocate state in VRAM\n");
        return -1;
    }
    
    /* Allocate guest RAM in VRAM */
    if (nearmem_alloc(nm_ctx, &ctx->memory.ram_region, ram_size) != NEARMEM_OK) {
        fprintf(stderr, "gpucpu: Failed to allocate RAM in VRAM\n");
        nearmem_free(nm_ctx, &ctx->memory.state_region);
        return -1;
    }
    
    /* Get CPU-accessible pointers (via BAR1 mmap) */
    ctx->state = (x86_state_t *)ctx->memory.state_region.cpu_ptr;
    ctx->ram = (uint8_t *)ctx->memory.ram_region.cpu_ptr;
    
    /* Clear state and RAM */
    memset(ctx->state, 0, sizeof(x86_state_t));
    memset(ctx->ram, 0, ram_size);
    
    /* Sync to GPU */
    nearmem_sync(nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
#ifdef NEARMEM_HAS_CUDA
    /* Initialize CUDA resources and enable GPU execution */
    if (gpucpu_cuda_available()) {
        /* Detect GPU tier and tiny mode */
        char gpu_name[256] = {0};
        size_t gpu_mem = 0;
        int major = 0, minor = 0, is_tiny = 0;
        
        if (gpucpu_cuda_get_device_info(0, gpu_name, sizeof(gpu_name),
                                         &gpu_mem, &major, &minor, &is_tiny) == 0) {
            int tier = gpucpu_cuda_get_tier(gpu_mem);
            
            /* Set GPU tier and tiny mode in context */
            ctx->gpu_tier = tier;
            ctx->tiny_mode = is_tiny ? true : false;
            
            const char *tier_names[] = {"NONE", "TINY", "LOW", "MID", "HIGH"};
            printf("gpucpu: GPU detected: %s\n", gpu_name);
            printf("gpucpu: VRAM: %zu MB, Compute: %d.%d, Tier: %s%s\n",
                   gpu_mem >> 20, major, minor,
                   tier_names[tier < 5 ? tier : 0],
                   is_tiny ? " (TINY MODE)" : "");
            
            if (is_tiny) {
                size_t recommended = gpucpu_cuda_get_recommended_ram(tier, gpu_mem);
                printf("gpucpu: Tiny mode enabled - recommended guest RAM: %zu MB\n",
                       recommended >> 20);
            }
        }
        
        if (gpucpu_cuda_init() == 0) {
            ctx->use_gpu = true;
            printf("gpucpu: GPU execution ENABLED (interpreter runs on GPU!)\n");
        } else {
            printf("gpucpu: GPU init failed, falling back to CPU\n");
            ctx->use_gpu = false;
            ctx->gpu_tier = 0;  /* GPU_TIER_NONE */
        }
    } else {
        printf("gpucpu: No CUDA device, using CPU execution\n");
        ctx->use_gpu = false;
        ctx->gpu_tier = 0;  /* GPU_TIER_NONE */
    }
#else
    ctx->use_gpu = false;
    ctx->gpu_tier = 0;  /* GPU_TIER_NONE */
    ctx->tiny_mode = false;
    printf("gpucpu: Compiled without CUDA, using CPU execution\n");
#endif
    
    printf("gpucpu: Initialized with %zu MB guest RAM in VRAM\n", ram_size >> 20);
    
    return 0;
}

void gpucpu_shutdown(gpucpu_ctx_t *ctx) {
    if (!ctx)
        return;
    
#ifdef NEARMEM_HAS_CUDA
    if (ctx->use_gpu) {
        gpucpu_cuda_shutdown();
    }
#endif
    
    if (ctx->memory.nm_ctx) {
        nearmem_free(ctx->memory.nm_ctx, &ctx->memory.state_region);
        nearmem_free(ctx->memory.nm_ctx, &ctx->memory.ram_region);
    }
    
    memset(ctx, 0, sizeof(*ctx));
}

void gpucpu_reset(gpucpu_ctx_t *ctx, int mode) {
    if (!ctx || !ctx->state)
        return;
    
    x86_state_t *s = ctx->state;
    
    memset(s, 0, sizeof(*s));
    
    /* Power-on reset state */
    s->rflags = 0x0002;  /* Reserved bit always 1 */
    
    if (mode == MODE_REAL) {
        /* Real mode: start at FFFF:0000 (typical BIOS entry) */
        /* For .COM files, we'll override to 0100h */
        s->cs = 0xFFFF;
        s->rip = 0x0000;
        s->ds = s->es = s->ss = 0x0000;
        s->rsp = 0xFFFE;  /* Stack at top of segment */
        ctx->mode = MODE_REAL;
    } else if (mode == MODE_LONG) {
        /* Long mode: 64-bit, flat memory */
        s->cr0 = CR0_PE | CR0_PG;
        s->efer = EFER_LME | EFER_LMA;
        s->cs = 0x08;     /* Code segment selector */
        s->ds = s->es = s->ss = 0x10;  /* Data segment selector */
        s->rip = 0;
        s->rsp = ctx->memory.ram_size - 8;
        ctx->mode = MODE_LONG;
    }
    
    /* Sync to GPU */
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
}

/*
 * ============================================================
 * Memory Access
 * ============================================================
 */

int gpucpu_load(gpucpu_ctx_t *ctx, const void *data, size_t size, uint64_t address) {
    if (!ctx || !ctx->ram)
        return -1;
    
    if (address + size > ctx->memory.ram_size) {
        fprintf(stderr, "gpucpu: Load exceeds RAM bounds\n");
        return -1;
    }
    
    memcpy(ctx->ram + address, data, size);
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
    
    return 0;
}

void gpucpu_set_entry(gpucpu_ctx_t *ctx, uint64_t rip) {
    if (ctx && ctx->state) {
        ctx->state->rip = rip;
        nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
    }
}

int gpucpu_read_mem(gpucpu_ctx_t *ctx, uint64_t addr, void *buf, size_t size) {
    if (!ctx || !ctx->ram || addr + size > ctx->memory.ram_size)
        return -1;
    
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
    memcpy(buf, ctx->ram + addr, size);
    return 0;
}

int gpucpu_write_mem(gpucpu_ctx_t *ctx, uint64_t addr, const void *buf, size_t size) {
    if (!ctx || !ctx->ram || addr + size > ctx->memory.ram_size)
        return -1;
    
    memcpy(ctx->ram + addr, buf, size);
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
    return 0;
}

/*
 * ============================================================
 * Register Access
 * ============================================================
 */

uint64_t gpucpu_get_reg(gpucpu_ctx_t *ctx, int reg) {
    if (!ctx || !ctx->state)
        return 0;
    
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    switch (reg) {
        case REG_RAX: return ctx->state->rax;
        case REG_RCX: return ctx->state->rcx;
        case REG_RDX: return ctx->state->rdx;
        case REG_RBX: return ctx->state->rbx;
        case REG_RSP: return ctx->state->rsp;
        case REG_RBP: return ctx->state->rbp;
        case REG_RSI: return ctx->state->rsi;
        case REG_RDI: return ctx->state->rdi;
        case REG_R8:  return ctx->state->r8;
        case REG_R9:  return ctx->state->r9;
        case REG_R10: return ctx->state->r10;
        case REG_R11: return ctx->state->r11;
        case REG_R12: return ctx->state->r12;
        case REG_R13: return ctx->state->r13;
        case REG_R14: return ctx->state->r14;
        case REG_R15: return ctx->state->r15;
        case REG_RIP: return ctx->state->rip;
        case REG_RFLAGS: return ctx->state->rflags;
        default: return 0;
    }
}

void gpucpu_set_reg(gpucpu_ctx_t *ctx, int reg, uint64_t value) {
    if (!ctx || !ctx->state)
        return;
    
    switch (reg) {
        case REG_RAX: ctx->state->rax = value; break;
        case REG_RCX: ctx->state->rcx = value; break;
        case REG_RDX: ctx->state->rdx = value; break;
        case REG_RBX: ctx->state->rbx = value; break;
        case REG_RSP: ctx->state->rsp = value; break;
        case REG_RBP: ctx->state->rbp = value; break;
        case REG_RSI: ctx->state->rsi = value; break;
        case REG_RDI: ctx->state->rdi = value; break;
        case REG_R8:  ctx->state->r8 = value; break;
        case REG_R9:  ctx->state->r9 = value; break;
        case REG_R10: ctx->state->r10 = value; break;
        case REG_R11: ctx->state->r11 = value; break;
        case REG_R12: ctx->state->r12 = value; break;
        case REG_R13: ctx->state->r13 = value; break;
        case REG_R14: ctx->state->r14 = value; break;
        case REG_R15: ctx->state->r15 = value; break;
        case REG_RIP: ctx->state->rip = value; break;
        case REG_RFLAGS: ctx->state->rflags = value; break;
    }
    
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
}

/*
 * ============================================================
 * 16-bit Real Mode Helper Functions
 * ============================================================
 */

/* Calculate linear address from segment:offset */
static inline uint32_t real_addr(x86_state_t *s, uint16_t seg, uint16_t off) {
    (void)s;  /* State not needed for linear address calculation */
    return ((uint32_t)seg << 4) + off;
}

/* Read byte from memory */
static inline uint8_t read8(gpucpu_ctx_t *ctx, uint32_t addr) {
    if (addr < ctx->memory.ram_size)
        return ctx->ram[addr];
    return 0xFF;
}

/* Read word from memory */
static inline uint16_t read16(gpucpu_ctx_t *ctx, uint32_t addr) {
    return read8(ctx, addr) | ((uint16_t)read8(ctx, addr + 1) << 8);
}

/* Write byte to memory */
static inline void write8(gpucpu_ctx_t *ctx, uint32_t addr, uint8_t val) {
    if (addr < ctx->memory.ram_size)
        ctx->ram[addr] = val;
}

/* Write word to memory */
static inline void write16(gpucpu_ctx_t *ctx, uint32_t addr, uint16_t val) {
    write8(ctx, addr, val & 0xFF);
    write8(ctx, addr + 1, val >> 8);
}

/* Fetch instruction byte and advance IP */
static inline uint8_t fetch8(gpucpu_ctx_t *ctx) {
    x86_state_t *s = ctx->state;
    uint32_t addr = real_addr(s, s->cs, s->rip & 0xFFFF);
    uint8_t val = read8(ctx, addr);
    s->rip++;
    return val;
}

/* Fetch instruction word and advance IP */
static inline uint16_t fetch16(gpucpu_ctx_t *ctx) {
    uint16_t lo = fetch8(ctx);
    uint16_t hi = fetch8(ctx);
    return lo | (hi << 8);
}

/* Push word onto stack */
static inline void push16(gpucpu_ctx_t *ctx, uint16_t val) {
    x86_state_t *s = ctx->state;
    s->rsp = (s->rsp - 2) & 0xFFFF;
    write16(ctx, real_addr(s, s->ss, s->rsp), val);
}

/* Pop word from stack */
static inline uint16_t pop16(gpucpu_ctx_t *ctx) {
    x86_state_t *s = ctx->state;
    uint16_t val = read16(ctx, real_addr(s, s->ss, s->rsp));
    s->rsp = (s->rsp + 2) & 0xFFFF;
    return val;
}

/* Update flags after arithmetic operation */
static void update_flags_8(x86_state_t *s, uint8_t result, int carry, int overflow) {
    s->rflags &= ~(FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_PF);
    
    if (carry) s->rflags |= FLAG_CF;
    if (overflow) s->rflags |= FLAG_OF;
    if (result == 0) s->rflags |= FLAG_ZF;
    if (result & 0x80) s->rflags |= FLAG_SF;
    
    /* Parity flag: set if even number of 1 bits in low byte */
    int bits = 0;
    for (int i = 0; i < 8; i++) if (result & (1 << i)) bits++;
    if ((bits & 1) == 0) s->rflags |= FLAG_PF;
}

static void update_flags_16(x86_state_t *s, uint16_t result, int carry, int overflow) {
    s->rflags &= ~(FLAG_CF | FLAG_ZF | FLAG_SF | FLAG_OF | FLAG_PF);
    
    if (carry) s->rflags |= FLAG_CF;
    if (overflow) s->rflags |= FLAG_OF;
    if (result == 0) s->rflags |= FLAG_ZF;
    if (result & 0x8000) s->rflags |= FLAG_SF;
    
    int bits = 0;
    for (int i = 0; i < 8; i++) if (result & (1 << i)) bits++;
    if ((bits & 1) == 0) s->rflags |= FLAG_PF;
}

/* Get register pointer by index (16-bit mode) */
static uint16_t *get_reg16(x86_state_t *s, int idx) {
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

static uint8_t *get_reg8(x86_state_t *s, int idx) {
    /* AL, CL, DL, BL, AH, CH, DH, BH */
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
 * Instruction Execution (16-bit Real Mode)
 * ============================================================
 */

int gpucpu_step(gpucpu_ctx_t *ctx) {
    if (!ctx || !ctx->state || !ctx->ram)
        return -1;
    
    x86_state_t *s = ctx->state;
    
    if (s->halted)
        return 0;
    
    /* Fetch opcode */
    uint8_t op = fetch8(ctx);
    
    /* Decode and execute */
    switch (op) {
        
        /* ===== NOP ===== */
        case 0x90:  /* NOP */
            break;
        
        /* ===== MOV immediate to register ===== */
        case 0xB0: case 0xB1: case 0xB2: case 0xB3:
        case 0xB4: case 0xB5: case 0xB6: case 0xB7:
            /* MOV r8, imm8 */
            *get_reg8(s, op - 0xB0) = fetch8(ctx);
            break;
        
        case 0xB8: case 0xB9: case 0xBA: case 0xBB:
        case 0xBC: case 0xBD: case 0xBE: case 0xBF:
            /* MOV r16, imm16 */
            *get_reg16(s, op - 0xB8) = fetch16(ctx);
            break;
        
        /* ===== MOV register to/from memory ===== */
        case 0x88: {  /* MOV r/m8, r8 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                /* Register to register */
                *get_reg8(s, rm) = *get_reg8(s, reg);
            } else {
                /* TODO: Memory addressing modes */
            }
            break;
        }
        
        case 0x89: {  /* MOV r/m16, r16 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                *get_reg16(s, rm) = *get_reg16(s, reg);
            }
            break;
        }
        
        case 0x8A: {  /* MOV r8, r/m8 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                *get_reg8(s, reg) = *get_reg8(s, rm);
            }
            break;
        }
        
        case 0x8B: {  /* MOV r16, r/m16 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                *get_reg16(s, reg) = *get_reg16(s, rm);
            }
            break;
        }
        
        /* ===== ADD ===== */
        case 0x00: {  /* ADD r/m8, r8 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint8_t a = *get_reg8(s, rm);
                uint8_t b = *get_reg8(s, reg);
                uint16_t result = a + b;
                *get_reg8(s, rm) = result;
                update_flags_8(s, result, result > 0xFF, 
                              ((a ^ result) & (b ^ result) & 0x80) != 0);
            }
            break;
        }
        
        case 0x01: {  /* ADD r/m16, r16 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint16_t a = *get_reg16(s, rm);
                uint16_t b = *get_reg16(s, reg);
                uint32_t result = a + b;
                *get_reg16(s, rm) = result;
                update_flags_16(s, result, result > 0xFFFF,
                               ((a ^ result) & (b ^ result) & 0x8000) != 0);
            }
            break;
        }
        
        case 0x04:  /* ADD AL, imm8 */
        {
            uint8_t imm = fetch8(ctx);
            uint8_t a = s->rax & 0xFF;
            uint16_t result = a + imm;
            s->rax = (s->rax & 0xFF00) | (result & 0xFF);
            update_flags_8(s, result, result > 0xFF,
                          ((a ^ result) & (imm ^ result) & 0x80) != 0);
            break;
        }
        
        case 0x05:  /* ADD AX, imm16 */
        {
            uint16_t imm = fetch16(ctx);
            uint16_t a = s->rax & 0xFFFF;
            uint32_t result = a + imm;
            s->rax = (s->rax & 0xFFFF0000) | (result & 0xFFFF);
            update_flags_16(s, result, result > 0xFFFF,
                           ((a ^ result) & (imm ^ result) & 0x8000) != 0);
            break;
        }
        
        /* ===== SUB ===== */
        case 0x28: {  /* SUB r/m8, r8 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint8_t a = *get_reg8(s, rm);
                uint8_t b = *get_reg8(s, reg);
                uint8_t result = a - b;
                *get_reg8(s, rm) = result;
                update_flags_8(s, result, a < b,
                              ((a ^ b) & (a ^ result) & 0x80) != 0);
            }
            break;
        }
        
        case 0x29: {  /* SUB r/m16, r16 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint16_t a = *get_reg16(s, rm);
                uint16_t b = *get_reg16(s, reg);
                uint16_t result = a - b;
                *get_reg16(s, rm) = result;
                update_flags_16(s, result, a < b,
                               ((a ^ b) & (a ^ result) & 0x8000) != 0);
            }
            break;
        }
        
        case 0x2C:  /* SUB AL, imm8 */
        {
            uint8_t imm = fetch8(ctx);
            uint8_t a = s->rax & 0xFF;
            uint8_t result = a - imm;
            s->rax = (s->rax & 0xFF00) | result;
            update_flags_8(s, result, a < imm,
                          ((a ^ imm) & (a ^ result) & 0x80) != 0);
            break;
        }
        
        /* ===== CMP ===== */
        case 0x38: {  /* CMP r/m8, r8 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint8_t a = *get_reg8(s, rm);
                uint8_t b = *get_reg8(s, reg);
                uint8_t result = a - b;
                update_flags_8(s, result, a < b,
                              ((a ^ b) & (a ^ result) & 0x80) != 0);
            }
            break;
        }
        
        case 0x39: {  /* CMP r/m16, r16 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint16_t a = *get_reg16(s, rm);
                uint16_t b = *get_reg16(s, reg);
                uint16_t result = a - b;
                update_flags_16(s, result, a < b,
                               ((a ^ b) & (a ^ result) & 0x8000) != 0);
            }
            break;
        }
        
        case 0x3C:  /* CMP AL, imm8 */
        {
            uint8_t imm = fetch8(ctx);
            uint8_t a = s->rax & 0xFF;
            uint8_t result = a - imm;
            update_flags_8(s, result, a < imm,
                          ((a ^ imm) & (a ^ result) & 0x80) != 0);
            break;
        }
        
        /* ===== INC/DEC ===== */
        case 0x40: case 0x41: case 0x42: case 0x43:
        case 0x44: case 0x45: case 0x46: case 0x47:
            /* INC r16 */
        {
            int reg = op - 0x40;
            uint16_t *r = get_reg16(s, reg);
            int old_cf = s->rflags & FLAG_CF;  /* Preserve CF */
            uint16_t result = *r + 1;
            *r = result;
            update_flags_16(s, result, 0, (*r == 0x8000));
            if (old_cf) s->rflags |= FLAG_CF;
            break;
        }
        
        case 0x48: case 0x49: case 0x4A: case 0x4B:
        case 0x4C: case 0x4D: case 0x4E: case 0x4F:
            /* DEC r16 */
        {
            int reg = op - 0x48;
            uint16_t *r = get_reg16(s, reg);
            int old_cf = s->rflags & FLAG_CF;
            uint16_t old_val = *r;
            uint16_t result = *r - 1;
            *r = result;
            update_flags_16(s, result, 0, (old_val == 0x8000));
            if (old_cf) s->rflags |= FLAG_CF;
            break;
        }
        
        /* ===== PUSH/POP ===== */
        case 0x50: case 0x51: case 0x52: case 0x53:
        case 0x54: case 0x55: case 0x56: case 0x57:
            /* PUSH r16 */
            push16(ctx, *get_reg16(s, op - 0x50));
            break;
        
        case 0x58: case 0x59: case 0x5A: case 0x5B:
        case 0x5C: case 0x5D: case 0x5E: case 0x5F:
            /* POP r16 */
            *get_reg16(s, op - 0x58) = pop16(ctx);
            break;
        
        /* ===== JMP ===== */
        case 0xEB:  /* JMP rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0xE9:  /* JMP rel16 */
        {
            int16_t rel = (int16_t)fetch16(ctx);
            s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        /* ===== Jcc (conditional jumps) ===== */
        case 0x74:  /* JE/JZ rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            if (s->rflags & FLAG_ZF)
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0x75:  /* JNE/JNZ rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            if (!(s->rflags & FLAG_ZF))
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0x7C:  /* JL rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            bool sf = (s->rflags & FLAG_SF) != 0;
            bool of = (s->rflags & FLAG_OF) != 0;
            if (sf != of)
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0x7D:  /* JGE rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            bool sf = (s->rflags & FLAG_SF) != 0;
            bool of = (s->rflags & FLAG_OF) != 0;
            if (sf == of)
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0x7E:  /* JLE rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            bool zf = (s->rflags & FLAG_ZF) != 0;
            bool sf = (s->rflags & FLAG_SF) != 0;
            bool of = (s->rflags & FLAG_OF) != 0;
            if (zf || (sf != of))
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0x7F:  /* JG rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            bool zf = (s->rflags & FLAG_ZF) != 0;
            bool sf = (s->rflags & FLAG_SF) != 0;
            bool of = (s->rflags & FLAG_OF) != 0;
            if (!zf && (sf == of))
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        /* ===== LOOP ===== */
        case 0xE2:  /* LOOP rel8 */
        {
            int8_t rel = (int8_t)fetch8(ctx);
            s->rcx = (s->rcx - 1) & 0xFFFF;
            if ((s->rcx & 0xFFFF) != 0)
                s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        /* ===== CALL/RET ===== */
        case 0xE8:  /* CALL rel16 */
        {
            int16_t rel = (int16_t)fetch16(ctx);
            push16(ctx, s->rip & 0xFFFF);
            s->rip = (s->rip + rel) & 0xFFFF;
            break;
        }
        
        case 0xC3:  /* RET */
            s->rip = pop16(ctx);
            break;
        
        /* ===== INT ===== */
        case 0xCD:  /* INT imm8 */
        {
            uint8_t vector = fetch8(ctx);
            
            /* Call interrupt handler if set */
            if (ctx->interrupt) {
                ctx->interrupt(ctx->int_ctx, vector);
            }
            
            /* Special handling for common interrupts */
            if (vector == 0x21) {
                /* DOS INT 21h - check AH for function */
                uint8_t ah = (s->rax >> 8) & 0xFF;
                
                if (ah == 0x4C) {
                    /* DOS terminate */
                    s->halted = true;
                }
                else if (ah == 0x02) {
                    /* DOS print character (DL) */
                    char c = s->rdx & 0xFF;
                    putchar(c);
                }
                else if (ah == 0x09) {
                    /* DOS print string at DS:DX, terminated by '$' */
                    uint32_t addr = real_addr(s, s->ds, s->rdx & 0xFFFF);
                    while (addr < ctx->memory.ram_size) {
                        char c = read8(ctx, addr++);
                        if (c == '$') break;
                        putchar(c);
                    }
                }
            }
            else if (vector == 0x20) {
                /* DOS terminate */
                s->halted = true;
            }
            break;
        }
        
        /* ===== HLT ===== */
        case 0xF4:
            s->halted = true;
            break;
        
        /* ===== XOR ===== */
        case 0x30: {  /* XOR r/m8, r8 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint8_t result = *get_reg8(s, rm) ^ *get_reg8(s, reg);
                *get_reg8(s, rm) = result;
                update_flags_8(s, result, 0, 0);
            }
            break;
        }
        
        case 0x31: {  /* XOR r/m16, r16 */
            uint8_t modrm = fetch8(ctx);
            int reg = (modrm >> 3) & 7;
            int rm = modrm & 7;
            if ((modrm & 0xC0) == 0xC0) {
                uint16_t result = *get_reg16(s, rm) ^ *get_reg16(s, reg);
                *get_reg16(s, rm) = result;
                update_flags_16(s, result, 0, 0);
            }
            break;
        }
        
        /* ===== Unknown opcode ===== */
        default:
            fprintf(stderr, "gpucpu: Unknown opcode %02X at %04X:%04X\n",
                    op, s->cs, (uint16_t)(s->rip - 1));
            s->halted = true;
            return -1;
    }
    
    s->instructions_executed++;
    return 1;
}

/*
 * ============================================================
 * Run Loop
 * ============================================================
 */

/*
 * Handle I/O interrupts that require host CPU involvement.
 * Called when GPU kernel returns with needs_io set.
 */
static void handle_io_interrupt(gpucpu_ctx_t *ctx, uint8_t vector) {
    x86_state_t *s = ctx->state;
    
    /* Sync state from GPU before reading */
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
    
    /* Skip the INT instruction (we backed up in kernel) */
    s->rip += 2;
    
    if (vector == 0x21) {
        /* DOS INT 21h */
        uint8_t ah = (s->rax >> 8) & 0xFF;
        
        if (ah == 0x4C) {
            /* DOS terminate */
            s->halted = true;
        }
        else if (ah == 0x02) {
            /* DOS print character (DL) */
            char c = s->rdx & 0xFF;
            putchar(c);
        }
        else if (ah == 0x09) {
            /* DOS print string at DS:DX, terminated by '$' */
            uint32_t addr = ((uint32_t)s->ds << 4) + (s->rdx & 0xFFFF);
            while (addr < ctx->memory.ram_size) {
                char c = ctx->ram[addr++];
                if (c == '$') break;
                putchar(c);
            }
        }
    }
    else if (vector == 0x20) {
        /* DOS terminate */
        s->halted = true;
    }
    
    /* Sync state back to GPU */
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_CPU_TO_GPU);
}

uint64_t gpucpu_run(gpucpu_ctx_t *ctx, uint64_t max_instructions) {
    if (!ctx || !ctx->state)
        return 0;
    
    double start_time = get_time_ms();
    uint64_t count = 0;
    
#ifdef NEARMEM_HAS_CUDA
    if (ctx->use_gpu) {
        /*
         * GPU EXECUTION PATH
         * ==================
         * The x86 interpreter runs entirely on GPU.
         * We only return to CPU for I/O interrupts (INT 21h).
         */
        while (!ctx->state->halted) {
            if (max_instructions > 0 && count >= max_instructions)
                break;
            
            int needs_io = 0;
            uint8_t io_vector = 0;
            uint64_t batch_max = (max_instructions > 0) ? 
                                 (max_instructions - count) : 0;
            
            /* Run on GPU until HLT or I/O needed */
            uint64_t executed = gpucpu_cuda_run(
                ctx->state,
                ctx->ram,
                ctx->memory.ram_size,
                batch_max,
                &needs_io,
                &io_vector
            );
            
            count += executed;
            
            if (needs_io) {
                /* Handle I/O on CPU, then continue on GPU */
                handle_io_interrupt(ctx, io_vector);
            } else {
                /* Sync final state from GPU */
                nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
                break;
            }
        }
    } else
#endif
    {
        /*
         * CPU EXECUTION PATH (fallback)
         * =============================
         * Traditional interpretation on host CPU.
         */
        while (!ctx->state->halted) {
            if (max_instructions > 0 && count >= max_instructions)
                break;
            
            int ret = gpucpu_step(ctx);
            if (ret < 0)
                break;
            
            count++;
        }
        
        /* Sync state back to CPU-accessible memory */
        nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
    }
    
    double elapsed = get_time_ms() - start_time;
    ctx->total_instructions += count;
    ctx->total_time_ms += elapsed;
    
    return count;
}

void gpucpu_interrupt(gpucpu_ctx_t *ctx, uint8_t vector) {
    if (ctx && ctx->state) {
        ctx->state->interrupt_pending = true;
        ctx->state->interrupt_vector = vector;
    }
}

/*
 * ============================================================
 * Debug/Utility Functions
 * ============================================================
 */

void gpucpu_dump_state(gpucpu_ctx_t *ctx) {
    if (!ctx || !ctx->state)
        return;
    
    nearmem_sync(ctx->memory.nm_ctx, NEARMEM_SYNC_GPU_TO_CPU);
    x86_state_t *s = ctx->state;
    
    printf("=== x86 State (via VRAM) ===\n");
    printf("AX=%04X  BX=%04X  CX=%04X  DX=%04X\n",
           (uint16_t)s->rax, (uint16_t)s->rbx,
           (uint16_t)s->rcx, (uint16_t)s->rdx);
    printf("SI=%04X  DI=%04X  BP=%04X  SP=%04X\n",
           (uint16_t)s->rsi, (uint16_t)s->rdi,
           (uint16_t)s->rbp, (uint16_t)s->rsp);
    printf("CS=%04X  DS=%04X  ES=%04X  SS=%04X\n",
           s->cs, s->ds, s->es, s->ss);
    printf("IP=%04X  FLAGS=%04X [%c%c%c%c%c%c]\n",
           (uint16_t)s->rip, (uint16_t)s->rflags,
           (s->rflags & FLAG_OF) ? 'O' : '-',
           (s->rflags & FLAG_SF) ? 'S' : '-',
           (s->rflags & FLAG_ZF) ? 'Z' : '-',
           (s->rflags & FLAG_CF) ? 'C' : '-',
           (s->rflags & FLAG_PF) ? 'P' : '-',
           (s->rflags & FLAG_AF) ? 'A' : '-');
    printf("Instructions: %lu\n", s->instructions_executed);
    
    if (ctx->total_time_ms > 0) {
        double mips = ctx->total_instructions / ctx->total_time_ms / 1000.0;
        printf("Performance: %.2f MIPS\n", mips);
    }
}
