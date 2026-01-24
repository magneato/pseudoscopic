/* SPDX-License-Identifier: GPL-2.0 */
/*
 * asm.h - Assembly function declarations
 *
 * Defines the interface between C and hand-optimized NASM routines.
 * These functions handle performance-critical memory operations.
 *
 * Copyright (C) 2025 Neural Splines LLC
 */

#ifndef _PSEUDOSCOPIC_ASM_H_
#define _PSEUDOSCOPIC_ASM_H_

#include <linux/types.h>

/*
 * Calling Convention: System V AMD64 ABI
 * --------------------------------------
 * Arguments: rdi, rsi, rdx, rcx, r8, r9
 * Return: rax
 * Callee-saved: rbx, rbp, r12-r15
 * Caller-saved: rax, rcx, rdx, rsi, rdi, r8-r11
 *
 * All functions are marked asmlinkage for explicit ABI.
 */

/*
 * Memory Copy Functions
 * ---------------------
 * Optimized for PCIe write-combining and streaming access.
 */

/*
 * ps_memcpy_to_vram - Copy from system RAM to VRAM
 * @dst:   Destination in VRAM (write-combining mapped)
 * @src:   Source in system RAM
 * @count: Bytes to copy (MUST be multiple of 64)
 *
 * Uses non-temporal stores (movntdq) to bypass cache on writes.
 * This achieves near-theoretical PCIe bandwidth by:
 *   - Avoiding cache pollution on destination
 *   - Coalescing writes into full cache lines
 *   - Minimizing read-for-ownership overhead
 *
 * Issues sfence after completion to ensure ordering.
 *
 * Context: May sleep (if source pages need faulting in).
 */
asmlinkage void ps_memcpy_to_vram(void __iomem *dst, 
                                   const void *src, 
                                   size_t count);

/*
 * ps_memcpy_from_vram - Copy from VRAM to system RAM
 * @dst:   Destination in system RAM
 * @src:   Source in VRAM (write-combining mapped)
 * @count: Bytes to copy (MUST be multiple of 64)
 *
 * Uses prefetch hints and temporal stores since we want
 * the data in cache for subsequent CPU access.
 *
 * WC-mapped VRAM reads are inherently uncached, so we
 * optimize for streaming reads with prefetchnta.
 *
 * Context: May sleep.
 */
asmlinkage void ps_memcpy_from_vram(void *dst, 
                                     const void __iomem *src, 
                                     size_t count);

/*
 * ps_memcpy_wc - Generic write-combining aware copy
 * @dst:   Destination address
 * @src:   Source address
 * @count: Bytes to copy (MUST be multiple of 64)
 * @to_wc: If true, use non-temporal stores; else temporal
 *
 * Unified copy routine that selects appropriate strategy.
 *
 * Context: May sleep.
 */
asmlinkage void ps_memcpy_wc(void *dst,
                              const void *src,
                              size_t count,
                              bool to_wc);

/*
 * Cache Control Functions
 * -----------------------
 * For explicit cache management when DMA is not an option.
 */

/*
 * ps_cache_flush - Flush and invalidate cache lines
 * @addr:  Start address (should be cache-line aligned)
 * @count: Bytes to flush (should be multiple of 64)
 *
 * Uses clflushopt (or clflush on older CPUs) to write back
 * dirty cache lines and invalidate. Issues sfence after.
 *
 * Use case: Ensure CPU writes are visible to DMA-capable device.
 *
 * Context: Atomic (no sleeping).
 */
asmlinkage void ps_cache_flush(void *addr, size_t count);

/*
 * ps_cache_writeback - Write back without invalidation
 * @addr:  Start address (should be cache-line aligned)
 * @count: Bytes to write back (should be multiple of 64)
 *
 * Uses clwb (or clflushopt as fallback) to write back dirty
 * cache lines while keeping them valid. More efficient when
 * CPU will access the data again soon.
 *
 * Use case: Ensure writes visible to device but keep cache hot.
 *
 * Context: Atomic.
 */
asmlinkage void ps_cache_writeback(void *addr, size_t count);

/*
 * ps_cache_invalidate - Invalidate cache lines
 * @addr:  Start address (should be cache-line aligned)
 * @count: Bytes to invalidate (should be multiple of 64)
 *
 * Discards cached data. Use after DMA to CPU to ensure
 * fresh reads. Currently implemented as clflush since
 * x86 lacks a pure invalidate instruction.
 *
 * Context: Atomic.
 */
asmlinkage void ps_cache_invalidate(void *addr, size_t count);

/*
 * Memory Barrier Functions
 * ------------------------
 * Explicit ordering primitives.
 */

/*
 * ps_sfence - Store fence
 *
 * Ensures all stores issued before the fence are globally
 * visible before any stores issued after. Required after
 * non-temporal stores.
 *
 * Context: Atomic.
 */
asmlinkage void ps_sfence(void);

/*
 * ps_lfence - Load fence
 *
 * Ensures all loads issued before the fence complete before
 * any loads issued after. Also serializes instruction stream.
 *
 * Context: Atomic.
 */
asmlinkage void ps_lfence(void);

/*
 * ps_mfence - Full memory fence
 *
 * Combines effects of sfence and lfence. Ensures all memory
 * operations complete before any subsequent operations.
 *
 * Context: Atomic.
 */
asmlinkage void ps_mfence(void);

/*
 * CPU Feature Detection
 * ---------------------
 * Query CPU capabilities for optimal code path selection.
 */

/*
 * ps_cpu_has_clflushopt - Check for CLFLUSHOPT support
 *
 * Returns: true if CLFLUSHOPT instruction is available
 *
 * CLFLUSHOPT is more efficient than CLFLUSH as it allows
 * parallel execution and better cache line handling.
 */
asmlinkage bool ps_cpu_has_clflushopt(void);

/*
 * ps_cpu_has_clwb - Check for CLWB support
 *
 * Returns: true if CLWB instruction is available
 *
 * CLWB (cache line write back) writes back without invalidate,
 * optimal for persistent memory patterns.
 */
asmlinkage bool ps_cpu_has_clwb(void);

/*
 * ps_cpu_has_avx - Check for AVX support
 *
 * Returns: true if AVX is available and enabled
 *
 * AVX provides 256-bit registers for wider copies.
 */
asmlinkage bool ps_cpu_has_avx(void);

/*
 * ps_cpu_has_avx512 - Check for AVX-512 support
 *
 * Returns: true if AVX-512 is available and enabled
 *
 * AVX-512 provides 512-bit registers and non-temporal hints.
 */
asmlinkage bool ps_cpu_has_avx512(void);

/*
 * Inline Helpers
 * --------------
 * For when function call overhead matters.
 */

/* Cache line size (assumed 64 bytes on modern x86) */
#define PS_CACHE_LINE_SIZE  64
#define PS_CACHE_LINE_MASK  (PS_CACHE_LINE_SIZE - 1)

/* Align address down to cache line boundary */
static inline void *ps_cache_align_down(void *addr)
{
    return (void *)((unsigned long)addr & ~PS_CACHE_LINE_MASK);
}

/* Align address up to cache line boundary */
static inline void *ps_cache_align_up(void *addr)
{
    return (void *)(((unsigned long)addr + PS_CACHE_LINE_MASK) & 
                    ~PS_CACHE_LINE_MASK);
}

/* Check if address is cache-line aligned */
static inline bool ps_is_cache_aligned(const void *addr)
{
    return ((unsigned long)addr & PS_CACHE_LINE_MASK) == 0;
}

/* Round size up to cache line multiple */
static inline size_t ps_cache_align_size(size_t size)
{
    return (size + PS_CACHE_LINE_MASK) & ~PS_CACHE_LINE_MASK;
}

#endif /* _PSEUDOSCOPIC_ASM_H_ */
