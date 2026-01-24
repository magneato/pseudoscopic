; SPDX-License-Identifier: GPL-2.0
;
; barriers.asm - Memory barrier primitives
;
; x86 has relatively strong memory ordering, but explicit
; barriers are still needed for:
;   - Non-temporal stores (require SFENCE)
;   - Device memory (may have weaker ordering)
;   - Coordination with other CPUs/devices
;
; These are thin wrappers that can be inlined, but having
; them as functions aids debugging and profiling.
;
; Copyright (C) 2025 Neural Splines LLC

BITS 64
DEFAULT REL

global ps_sfence:function
global ps_lfence:function
global ps_mfence:function

section .text

;-----------------------------------------------------------------------------
; ps_sfence - Store fence
;
; Ensures all stores before the fence are globally visible
; before any stores after the fence. Required after non-temporal
; stores (movntdq, movntps, etc.) to guarantee ordering.
;
; Cost: ~10-30 cycles depending on store buffer state
;-----------------------------------------------------------------------------
ps_sfence:
    sfence
    ret


;-----------------------------------------------------------------------------
; ps_lfence - Load fence
;
; Ensures all loads before the fence complete before any loads
; after. Also serializes instruction execution (useful for
; speculation control).
;
; Note: On modern Intel, LFENCE is dispatch-serializing,
; meaning it also prevents speculative execution across it.
;
; Cost: ~5-10 cycles
;-----------------------------------------------------------------------------
ps_lfence:
    lfence
    ret


;-----------------------------------------------------------------------------
; ps_mfence - Full memory fence
;
; Combines SFENCE and LFENCE - ensures all memory operations
; before complete before any after. The nuclear option.
;
; Use sparingly - this is expensive (~30-50 cycles).
;
; Needed when:
;   - Synchronizing with device memory that may reorder
;   - Implementing acquire/release semantics manually
;   - When in doubt about ordering requirements
;-----------------------------------------------------------------------------
ps_mfence:
    mfence
    ret
