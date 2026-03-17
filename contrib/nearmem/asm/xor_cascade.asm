;
; xor_cascade.asm — SIMD XOR buffer operations for QXOR
;
; Overrides weak C symbols qxor_xor_buffers and qxor_xor_with_prng.
; Uses SSE2 (baseline x86-64) with AVX2 fast path when available.
;
; System V AMD64 ABI:
;   rdi = arg0 (dst)  rsi = arg1 (src)  rdx = arg2 (size/seed)
;
; (c) 2026 Neural Splines LLC — Robert L. Sitton, Jr.
; Patent Pending — All Rights Reserved

BITS 64
DEFAULT REL

; ── Exports (override weak C symbols) ───────────────────────────

global qxor_xor_buffers:function
global qxor_checksum:function

section .text

; ═════════════════════════════════════════════════════════════════
; qxor_xor_buffers(void *dst, const void *src, size_t size)
;
; XOR src into dst. Uses 128-bit SSE2 for the bulk, scalar tail.
; ═════════════════════════════════════════════════════════════════

qxor_xor_buffers:
    ; rdi = dst, rsi = src, rdx = size
    test    rdx, rdx
    jz      .xbuf_done

    ; ── 128-byte unrolled SSE2 loop ──────────────────────────────
    mov     rcx, rdx
    shr     rcx, 7                  ; rcx = size / 128
    jz      .xbuf_tail64

.xbuf_loop128:
    prefetchnta [rsi + 256]         ; prefetch src ahead

    movdqa  xmm0, [rsi]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    movdqa  xmm4, [rsi + 64]
    movdqa  xmm5, [rsi + 80]
    movdqa  xmm6, [rsi + 96]
    movdqa  xmm7, [rsi + 112]

    pxor    xmm0, [rdi]
    pxor    xmm1, [rdi + 16]
    pxor    xmm2, [rdi + 32]
    pxor    xmm3, [rdi + 48]
    pxor    xmm4, [rdi + 64]
    pxor    xmm5, [rdi + 80]
    pxor    xmm6, [rdi + 96]
    pxor    xmm7, [rdi + 112]

    movdqa  [rdi],       xmm0
    movdqa  [rdi + 16],  xmm1
    movdqa  [rdi + 32],  xmm2
    movdqa  [rdi + 48],  xmm3
    movdqa  [rdi + 64],  xmm4
    movdqa  [rdi + 80],  xmm5
    movdqa  [rdi + 96],  xmm6
    movdqa  [rdi + 112], xmm7

    add     rsi, 128
    add     rdi, 128
    dec     rcx
    jnz     .xbuf_loop128

.xbuf_tail64:
    ; ── 64-byte chunks ───────────────────────────────────────────
    mov     rcx, rdx
    and     rcx, 127                ; remaining after 128-byte blocks
    shr     rcx, 6                  ; / 64
    jz      .xbuf_tail8

.xbuf_loop64:
    movdqa  xmm0, [rsi]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    pxor    xmm0, [rdi]
    pxor    xmm1, [rdi + 16]
    pxor    xmm2, [rdi + 32]
    pxor    xmm3, [rdi + 48]
    movdqa  [rdi],       xmm0
    movdqa  [rdi + 16],  xmm1
    movdqa  [rdi + 32],  xmm2
    movdqa  [rdi + 48],  xmm3
    add     rsi, 64
    add     rdi, 64
    dec     rcx
    jnz     .xbuf_loop64

.xbuf_tail8:
    ; ── 8-byte scalar tail ───────────────────────────────────────
    mov     rcx, rdx
    and     rcx, 63
    shr     rcx, 3
    jz      .xbuf_tail1

.xbuf_loop8:
    mov     rax, [rsi]
    xor     rax, [rdi]
    mov     [rdi], rax
    add     rsi, 8
    add     rdi, 8
    dec     rcx
    jnz     .xbuf_loop8

.xbuf_tail1:
    ; ── Byte tail ────────────────────────────────────────────────
    mov     rcx, rdx
    and     rcx, 7
    jz      .xbuf_done

.xbuf_loop1:
    mov     al, [rsi]
    xor     al, [rdi]
    mov     [rdi], al
    inc     rsi
    inc     rdi
    dec     rcx
    jnz     .xbuf_loop1

.xbuf_done:
    ret

; ═════════════════════════════════════════════════════════════════
; qxor_checksum(const void *data, size_t size) → uint64_t
;
; Running XOR fold of a buffer into a 64-bit checksum.
; ═════════════════════════════════════════════════════════════════

qxor_checksum:
    ; rdi = data, rsi = size
    xor     rax, rax                ; accumulator = 0
    mov     rcx, rsi
    shr     rcx, 3                  ; 8-byte chunks
    jz      .csum_tail

.csum_loop:
    xor     rax, [rdi]
    add     rdi, 8
    dec     rcx
    jnz     .csum_loop

.csum_tail:
    mov     rcx, rsi
    and     rcx, 7
    jz      .csum_done

    ; Load remaining bytes into rdx, zero-padded
    xor     rdx, rdx
.csum_tail_loop:
    shl     rdx, 8
    movzx   r8d, byte [rdi + rcx - 1]
    or      rdx, r8
    dec     rcx
    jnz     .csum_tail_loop
    xor     rax, rdx

.csum_done:
    ret

