; SPDX-License-Identifier: GPL-2.0
;
; cache_ops.asm - Cache line management operations
;
; Provides wrappers for x86 cache control instructions with
; automatic fallback for older CPUs. Essential for CPU-GPU
; coherency when DMA is not available.
;
; Instruction availability:
;   CLFLUSH    - Pentium 4+ (virtually universal)
;   CLFLUSHOPT - Skylake+ (more efficient, allows parallel execution)
;   CLWB       - Skylake-X, Ice Lake+ (writeback without invalidate)
;
; Copyright (C) 2026 Neural Splines LLC

BITS 64
DEFAULT REL

; Exports
global ps_cache_flush:function
global ps_cache_writeback:function
global ps_cache_invalidate:function
global ps_cpu_has_clflushopt:function
global ps_cpu_has_clwb:function
global ps_cpu_has_avx:function
global ps_cpu_has_avx512:function

; CPUID feature bits
%define CPUID_CLFLUSHOPT_BIT    23      ; EBX bit for CLFLUSHOPT (leaf 7)
%define CPUID_CLWB_BIT          24      ; EBX bit for CLWB (leaf 7)
%define CPUID_AVX_BIT           28      ; ECX bit for AVX (leaf 1)
%define CPUID_AVX512F_BIT       16      ; EBX bit for AVX-512F (leaf 7)

%define CACHE_LINE_SIZE         64

section .data
    ; Feature flags cached on first call
    align 8
    features_detected:  dq 0    ; 0 = not yet, 1 = detected
    has_clflushopt:     db 0
    has_clwb:           db 0
    has_avx:            db 0
    has_avx512:         db 0

section .text

;-----------------------------------------------------------------------------
; detect_features - One-time CPU feature detection
;
; Called internally to populate feature flags.
; Thread-safe via simple flag check (races are benign).
;-----------------------------------------------------------------------------
detect_features:
    ; Check if already detected
    mov     rax, [rel features_detected]
    test    rax, rax
    jnz     .done
    
    push    rbx
    push    rcx
    push    rdx
    
    ; CPUID leaf 0: Get max supported leaf
    xor     eax, eax
    cpuid
    cmp     eax, 7
    jb      .no_leaf7
    
    ; CPUID leaf 1: Basic features (AVX)
    mov     eax, 1
    cpuid
    test    ecx, (1 << CPUID_AVX_BIT)
    setnz   al
    mov     [rel has_avx], al
    
    ; CPUID leaf 7, subleaf 0: Extended features
    mov     eax, 7
    xor     ecx, ecx
    cpuid
    
    ; CLFLUSHOPT: EBX bit 23
    test    ebx, (1 << CPUID_CLFLUSHOPT_BIT)
    setnz   al
    mov     [rel has_clflushopt], al
    
    ; CLWB: EBX bit 24
    test    ebx, (1 << CPUID_CLWB_BIT)
    setnz   al
    mov     [rel has_clwb], al
    
    ; AVX-512F: EBX bit 16
    test    ebx, (1 << CPUID_AVX512F_BIT)
    setnz   al
    mov     [rel has_avx512], al
    
    jmp     .finish
    
.no_leaf7:
    ; Old CPU without extended features
    mov     byte [rel has_clflushopt], 0
    mov     byte [rel has_clwb], 0
    mov     byte [rel has_avx512], 0
    
    ; Still check AVX via leaf 1
    mov     eax, 1
    cpuid
    test    ecx, (1 << CPUID_AVX_BIT)
    setnz   al
    mov     [rel has_avx], al
    
.finish:
    ; Mark as detected
    mov     qword [rel features_detected], 1
    
    pop     rdx
    pop     rcx
    pop     rbx
    
.done:
    ret


;-----------------------------------------------------------------------------
; ps_cache_flush - Flush and invalidate cache lines
;
; void ps_cache_flush(void *addr, size_t count)
;
; Arguments:
;   rdi = addr  - Start address
;   rdx = count - Bytes to flush
;
; Uses CLFLUSHOPT if available, otherwise CLFLUSH.
; Issues SFENCE after to ensure completion.
;-----------------------------------------------------------------------------
ps_cache_flush:
    push    rbx
    
    ; Ensure we know CPU features
    call    detect_features
    
    ; Handle zero-length
    test    rsi, rsi
    jz      .done
    
    ; Align start address down to cache line
    mov     rax, rdi
    and     rdi, ~(CACHE_LINE_SIZE - 1)
    
    ; Calculate number of cache lines
    add     rsi, rax            ; Add misalignment to count
    sub     rsi, rdi
    add     rsi, CACHE_LINE_SIZE - 1
    shr     rsi, 6              ; rsi = number of cache lines
    
    ; Select instruction based on CPU support
    cmp     byte [rel has_clflushopt], 0
    jz      .use_clflush
    
    ; Use CLFLUSHOPT (more efficient)
.loop_clflushopt:
    clflushopt [rdi]
    add     rdi, CACHE_LINE_SIZE
    dec     rsi
    jnz     .loop_clflushopt
    jmp     .fence
    
    ; Fallback to CLFLUSH
.use_clflush:
.loop_clflush:
    clflush [rdi]
    add     rdi, CACHE_LINE_SIZE
    dec     rsi
    jnz     .loop_clflush
    
.fence:
    sfence                      ; Ensure flushes complete
    
.done:
    pop     rbx
    ret


;-----------------------------------------------------------------------------
; ps_cache_writeback - Write back cache lines without invalidation
;
; void ps_cache_writeback(void *addr, size_t count)
;
; Arguments:
;   rdi = addr  - Start address
;   rsi = count - Bytes to write back
;
; Uses CLWB if available, otherwise falls back to CLFLUSHOPT/CLFLUSH.
; CLWB is optimal for persistent memory patterns where CPU will
; access data again.
;-----------------------------------------------------------------------------
ps_cache_writeback:
    push    rbx
    
    call    detect_features
    
    test    rsi, rsi
    jz      .done
    
    ; Align and calculate cache lines
    mov     rax, rdi
    and     rdi, ~(CACHE_LINE_SIZE - 1)
    add     rsi, rax
    sub     rsi, rdi
    add     rsi, CACHE_LINE_SIZE - 1
    shr     rsi, 6
    
    ; Try CLWB first (best)
    cmp     byte [rel has_clwb], 0
    jz      .try_clflushopt
    
.loop_clwb:
    clwb    [rdi]
    add     rdi, CACHE_LINE_SIZE
    dec     rsi
    jnz     .loop_clwb
    jmp     .fence
    
.try_clflushopt:
    cmp     byte [rel has_clflushopt], 0
    jz      .use_clflush
    
.loop_clflushopt:
    clflushopt [rdi]
    add     rdi, CACHE_LINE_SIZE
    dec     rsi
    jnz     .loop_clflushopt
    jmp     .fence
    
.use_clflush:
.loop_clflush:
    clflush [rdi]
    add     rdi, CACHE_LINE_SIZE
    dec     rsi
    jnz     .loop_clflush
    
.fence:
    sfence
    
.done:
    pop     rbx
    ret


;-----------------------------------------------------------------------------
; ps_cache_invalidate - Invalidate cache lines
;
; void ps_cache_invalidate(void *addr, size_t count)
;
; Arguments:
;   rdi = addr  - Start address
;   rsi = count - Bytes to invalidate
;
; x86 lacks a pure invalidate instruction, so this uses CLFLUSH
; which writes back then invalidates. Equivalent for our use case
; (ensuring fresh reads after DMA).
;-----------------------------------------------------------------------------
ps_cache_invalidate:
    ; Same as flush - x86 doesn't have pure invalidate
    jmp     ps_cache_flush


;-----------------------------------------------------------------------------
; Feature Query Functions
;-----------------------------------------------------------------------------

ps_cpu_has_clflushopt:
    call    detect_features
    movzx   eax, byte [rel has_clflushopt]
    ret

ps_cpu_has_clwb:
    call    detect_features
    movzx   eax, byte [rel has_clwb]
    ret

ps_cpu_has_avx:
    call    detect_features
    movzx   eax, byte [rel has_avx]
    ret

ps_cpu_has_avx512:
    call    detect_features
    movzx   eax, byte [rel has_avx512]
    ret
