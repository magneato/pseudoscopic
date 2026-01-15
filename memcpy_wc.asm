; SPDX-License-Identifier: GPL-2.0
;
; memcpy_wc.asm - Write-combining optimized memory copy
;
; Hand-tuned NASM for maximum PCIe bandwidth when copying
; between system RAM and write-combining mapped VRAM.
;
; The key insight: PCIe writes are posted (fire-and-forget),
; but reads require round-trips. Non-temporal stores bypass
; the cache hierarchy entirely, eliminating read-for-ownership
; overhead and cache pollution.
;
; Copyright (C) 2025 Neural Splines LLC
; Author: Robert L. Sitton, Jr.


BITS 64
DEFAULT REL

; Export symbols for kernel module linking
global ps_memcpy_to_vram:function
global ps_memcpy_from_vram:function
global ps_memcpy_wc:function

; Constants
%define CACHE_LINE_SIZE     64
%define UNROLL_FACTOR       4           ; Process 4 cache lines per iteration
%define BYTES_PER_ITER      (CACHE_LINE_SIZE * UNROLL_FACTOR)

section .text

;-----------------------------------------------------------------------------
; ps_memcpy_to_vram - Copy from system RAM to VRAM
;
; void ps_memcpy_to_vram(void *dst, const void *src, size_t count)
;
; Arguments (System V AMD64 ABI):
;   rdi = dst   - Destination in VRAM (write-combining mapped)
;   rsi = src   - Source in system RAM
;   rdx = count - Bytes to copy (must be multiple of 64)
;
; Uses non-temporal stores to:
;   1. Bypass cache (no pollution)
;   2. Coalesce writes into full cache lines
;   3. Avoid read-for-ownership latency
;
; Achieves ~12-14 GB/s on PCIe Gen3 x16
;-----------------------------------------------------------------------------
ps_memcpy_to_vram:
    ; Preserve callee-saved registers we'll use
    push    rbx
    
    ; Handle zero-length copy
    test    rdx, rdx
    jz      .done
    
    ; Main loop: 256 bytes per iteration (4 cache lines)
    ; This balances instruction overhead vs. throughput
    mov     rcx, rdx
    shr     rcx, 8              ; rcx = count / 256
    jz      .tail_64            ; Less than 256 bytes
    
.loop_256:
    ; Prefetch source data for next iteration
    ; Distance tuned for PCIe + memory latency hiding
    prefetchnta [rsi + 512]
    prefetchnta [rsi + 576]
    prefetchnta [rsi + 640]
    prefetchnta [rsi + 704]
    
    ; Cache line 0: bytes 0-63
    movdqa  xmm0, [rsi]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    
    movntdq [rdi], xmm0
    movntdq [rdi + 16], xmm1
    movntdq [rdi + 32], xmm2
    movntdq [rdi + 48], xmm3
    
    ; Cache line 1: bytes 64-127
    movdqa  xmm4, [rsi + 64]
    movdqa  xmm5, [rsi + 80]
    movdqa  xmm6, [rsi + 96]
    movdqa  xmm7, [rsi + 112]
    
    movntdq [rdi + 64], xmm4
    movntdq [rdi + 80], xmm5
    movntdq [rdi + 96], xmm6
    movntdq [rdi + 112], xmm7
    
    ; Cache line 2: bytes 128-191
    movdqa  xmm0, [rsi + 128]
    movdqa  xmm1, [rsi + 144]
    movdqa  xmm2, [rsi + 160]
    movdqa  xmm3, [rsi + 176]
    
    movntdq [rdi + 128], xmm0
    movntdq [rdi + 144], xmm1
    movntdq [rdi + 160], xmm2
    movntdq [rdi + 176], xmm3
    
    ; Cache line 3: bytes 192-255
    movdqa  xmm4, [rsi + 192]
    movdqa  xmm5, [rsi + 208]
    movdqa  xmm6, [rsi + 224]
    movdqa  xmm7, [rsi + 240]
    
    movntdq [rdi + 192], xmm4
    movntdq [rdi + 208], xmm5
    movntdq [rdi + 224], xmm6
    movntdq [rdi + 240], xmm7
    
    ; Advance pointers
    add     rsi, 256
    add     rdi, 256
    dec     rcx
    jnz     .loop_256
    
    ; Handle remaining 64-byte chunks
    and     rdx, 255            ; Remaining bytes after 256-byte blocks
    
.tail_64:
    mov     rcx, rdx
    shr     rcx, 6              ; rcx = remaining / 64
    jz      .finish
    
.loop_64:
    movdqa  xmm0, [rsi]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    
    movntdq [rdi], xmm0
    movntdq [rdi + 16], xmm1
    movntdq [rdi + 32], xmm2
    movntdq [rdi + 48], xmm3
    
    add     rsi, 64
    add     rdi, 64
    dec     rcx
    jnz     .loop_64
    
.finish:
    ; Ensure all non-temporal stores complete before returning
    ; This is critical for correct ordering with subsequent operations
    sfence
    
.done:
    pop     rbx
    ret


;-----------------------------------------------------------------------------
; ps_memcpy_from_vram - Copy from VRAM to system RAM
;
; void ps_memcpy_from_vram(void *dst, const void *src, size_t count)
;
; Arguments:
;   rdi = dst   - Destination in system RAM
;   rsi = src   - Source in VRAM (write-combining mapped)
;   rdx = count - Bytes to copy (must be multiple of 64)
;
; VRAM reads through WC mapping are inherently uncached.
; We use temporal stores since the CPU likely needs the data.
;
; Note: Read bandwidth from VRAM is limited by PCIe and the
; GPU's memory controller response time. Prefetching has
; limited benefit but doesn't hurt.
;-----------------------------------------------------------------------------
ps_memcpy_from_vram:
    push    rbx
    
    test    rdx, rdx
    jz      .done
    
    ; Main loop: 256 bytes per iteration
    mov     rcx, rdx
    shr     rcx, 8
    jz      .tail_64
    
.loop_256:
    ; Prefetch hint (limited benefit for MMIO reads, but harmless)
    prefetchnta [rsi + 512]
    
    ; Cache line 0
    movdqa  xmm0, [rsi]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    
    ; Temporal stores - we want this in cache
    movdqa  [rdi], xmm0
    movdqa  [rdi + 16], xmm1
    movdqa  [rdi + 32], xmm2
    movdqa  [rdi + 48], xmm3
    
    ; Cache line 1
    movdqa  xmm4, [rsi + 64]
    movdqa  xmm5, [rsi + 80]
    movdqa  xmm6, [rsi + 96]
    movdqa  xmm7, [rsi + 112]
    
    movdqa  [rdi + 64], xmm4
    movdqa  [rdi + 80], xmm5
    movdqa  [rdi + 96], xmm6
    movdqa  [rdi + 112], xmm7
    
    ; Cache line 2
    movdqa  xmm0, [rsi + 128]
    movdqa  xmm1, [rsi + 144]
    movdqa  xmm2, [rsi + 160]
    movdqa  xmm3, [rsi + 176]
    
    movdqa  [rdi + 128], xmm0
    movdqa  [rdi + 144], xmm1
    movdqa  [rdi + 160], xmm2
    movdqa  [rdi + 176], xmm3
    
    ; Cache line 3
    movdqa  xmm4, [rsi + 192]
    movdqa  xmm5, [rsi + 208]
    movdqa  xmm6, [rsi + 224]
    movdqa  xmm7, [rsi + 240]
    
    movdqa  [rdi + 192], xmm4
    movdqa  [rdi + 208], xmm5
    movdqa  [rdi + 224], xmm6
    movdqa  [rdi + 240], xmm7
    
    add     rsi, 256
    add     rdi, 256
    dec     rcx
    jnz     .loop_256
    
    and     rdx, 255
    
.tail_64:
    mov     rcx, rdx
    shr     rcx, 6
    jz      .done
    
.loop_64:
    movdqa  xmm0, [rsi]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    
    movdqa  [rdi], xmm0
    movdqa  [rdi + 16], xmm1
    movdqa  [rdi + 32], xmm2
    movdqa  [rdi + 48], xmm3
    
    add     rsi, 64
    add     rdi, 64
    dec     rcx
    jnz     .loop_64
    
.done:
    ; No fence needed - temporal stores are ordered
    pop     rbx
    ret


;-----------------------------------------------------------------------------
; ps_memcpy_wc - Generic write-combining aware copy
;
; void ps_memcpy_wc(void *dst, const void *src, size_t count, bool to_wc)
;
; Arguments:
;   rdi = dst   - Destination address
;   rsi = src   - Source address
;   rdx = count - Bytes to copy (must be multiple of 64)
;   rcx = to_wc - If true, use non-temporal stores
;
; Unified entry point that dispatches to appropriate implementation.
;-----------------------------------------------------------------------------
ps_memcpy_wc:
    test    cl, cl
    jz      ps_memcpy_from_vram     ; to_wc == false: temporal stores
    jmp     ps_memcpy_to_vram       ; to_wc == true: non-temporal stores
