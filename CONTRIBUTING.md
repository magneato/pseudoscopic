```
    ╔═══════════════════════════════════════════════════════════════════════════════════╗
    ║                                                                                   ║
    ║   ██████╗ ██████╗ ███╗   ██╗████████╗██████╗ ██╗██████╗ ██╗   ██╗████████╗███████╗║
    ║  ██╔════╝██╔═══██╗████╗  ██║╚══██╔══╝██╔══██╗██║██╔══██╗██║   ██║╚══██╔══╝██╔════╝║
    ║  ██║     ██║   ██║██╔██╗ ██║   ██║   ██████╔╝██║██████╔╝██║   ██║   ██║   █████╗  ║
    ║  ██║     ██║   ██║██║╚██╗██║   ██║   ██╔══██╗██║██╔══██╗██║   ██║   ██║   ██╔══╝  ║
    ║  ╚██████╗╚██████╔╝██║ ╚████║   ██║   ██║  ██║██║██████╔╝╚██████╔╝   ██║   ███████╗║
    ║   ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═╝╚═╝╚═════╝  ╚═════╝    ╚═╝   ╚══════╝║
    ║                                                                                   ║
    ║               Guidelines for Those Who Wish to Shape the Future                   ║
    ╚═══════════════════════════════════════════════════════════════════════════════════╝
```

---

## ◈ A Preface on Philosophy

Thank you for considering contributing to Pseudoscopic. Before you dive into code, understand that this project embodies a particular philosophy—one of *minimal elegance*.

Every line of code must earn its place. Like a haiku, we seek to express maximum meaning with minimum syntax. The best contribution is often the one that makes the codebase *smaller* while increasing capability.

> **Historical note**: The original UNIX philosophy—"do one thing well"—emerged from Bell Labs in the 1970s when memory was measured in kilobytes. We've lost that discipline in an age of gigabytes. Pseudoscopic is an attempt to recover it.

---

## ◈ The Four Pillars

```
    ┌─────────────────────────────────────────────────────────────────────────────┐
    │                       PSEUDOSCOPIC CODE PHILOSOPHY                          │
    ├─────────────────────────────────────────────────────────────────────────────┤
    │                                                                             │
    │   ╔═══════════════╗    ╔═══════════════╗    ╔═══════════════╗               │
    │   ║   MINIMAL     ║    ║  BULLETPROOF  ║    ║   BEAUTIFUL   ║               │
    │   ║   SURFACE     ║    ║  RELIABILITY  ║    ║   IN CODE     ║               │
    │   ║   AREA        ║    ║               ║    ║               ║               │
    │   ╚═══════════════╝    ╚═══════════════╝    ╚═══════════════╝               │
    │          │                    │                    │                        │
    │          ▼                    ▼                    ▼                        │
    │   No feature bloat     Every error path      If it's not                    │
    │   Every API call       is handled            readable, it's                 │
    │   justified            Lock-free is not      not maintainable               │
    │                        lockless              Comments explain               │
    │                                               *why*, not *what*             │
    │                                                                             │
    │                       ╔═══════════════════════════╗                         │
    │                       ║   PERFORMANCE WHERE       ║                         │
    │                       ║   IT MATTERS              ║                         │
    │                       ╚═══════════════════════════╝                         │
    │                                  │                                          │
    │                                  ▼                                          │
    │                       Assembly for hot paths                                │
    │                       Clarity for cold paths                                │
    │                       Measure before optimize                               │
    │                                                                             │
    └─────────────────────────────────────────────────────────────────────────────┘
```

---

## ◈ Code of Conduct

Be excellent to each other. That's the whole code.

This is a technical project; keep discussions technical. We evaluate code, not people. Critique is welcome; condescension is not.

> **On disagreement**: The best technical arguments are won with evidence, not volume. If you believe a different approach is superior, write the code. Show the benchmarks. The repository is the arena.

---

## ◈ Setting Up Your Forge

### Development Environment

```bash
# Clone your fork
git clone https://github.com/magneato/pseudoscopic.git
cd pseudoscopic

# Install the tools of the trade
sudo apt install build-essential linux-headers-$(uname -r) nasm \
    sparse cppcheck clang-format

# Build with debug symbols (you'll want these)
make DEBUG=1

# Run static analysis (find bugs before they find you)
make check
```

### Understanding the Territory

```
    pseudoscopic/
    ├── src/
    │   ├── core/          ◀─── Module entry, PCI ceremony, memory pools
    │   ├── hmm/           ◀─── Heterogeneous Memory Management magic
    │   ├── dma/           ◀─── Direct Memory Access engine wrangling  
    │   └── asm/           ◀─── Hand-optimized NASM (the hot paths)
    ├── include/           ◀─── Public headers (the contract)
    ├── contrib/
    │   └── nearmem/       ◀─── Userspace library (where most work happens)
    ├── nmc/               ◀─── Near-Memory Computing primitives
    ├── tools/             ◀─── Userspace utilities
    └── scripts/           ◀─── Installation and helper scripts
```

---

## ◈ The Art of the Change

### Before You Write a Single Line

1. **Search existing issues** — Your brilliant idea may already be in progress
2. **Open an issue first** for significant changes — Discussion prevents wasted effort
3. **One PR, one purpose** — Atomic changes are reviewable changes

> **A cautionary tale**: The Linux kernel once received a 50,000-line patch. It was rejected not because it was wrong, but because no human could review it. Linus Torvalds said: "Make each patch do ONE thing." We agree.

### Coding Standards for C

We follow the [Linux kernel coding style](https://www.kernel.org/doc/html/latest/process/coding-style.html). Not because it's perfect, but because consistency matters more than perfection.

```c
/*
 * Good: Clear purpose, error handling, single responsibility
 */
static int ps_pool_alloc_page(struct ps_pool *pool, struct page **page_out)
{
    struct page *page;
    unsigned long flags;
    
    /* Lock first, think later (but not much later) */
    spin_lock_irqsave(&pool->lock, flags);
    
    page = pool->free_list;
    if (!page) {
        spin_unlock_irqrestore(&pool->lock, flags);
        return -ENOMEM;  /* Memory pressure is not a crime */
    }
    
    pool->free_list = page->zone_device_data;
    atomic_long_dec(&pool->free);
    
    spin_unlock_irqrestore(&pool->lock, flags);
    
    *page_out = page;
    return 0;
}
```

**Key stylistic elements:**
- **Tabs for indentation**, displayed at 8 spaces (the kernel way)
- **80-column soft limit** — Your code will be read on terminals from 2003
- **Braces**: Same line for control flow, own line for functions
- **Check every return value** — Errors don't simply vanish
- **Error paths clean up** — Memory leaks are crimes against humanity

### Coding Standards for Assembly (NASM)

Assembly is where performance lives. Document it like your life depends on it—because someone's debugging session might.

```nasm
;═══════════════════════════════════════════════════════════════════════════════
; ps_memcpy_to_vram - Copy from system RAM to GPU VRAM
;
; This function implements write-combining optimized bulk transfer. It uses
; non-temporal stores (MOVNTDQ) to bypass CPU cache, ensuring data flows
; directly to the PCIe bus without polluting L1/L2/L3.
;
; Historical context: Non-temporal hints were introduced with SSE (1999) to
; handle graphics card uploads. Twenty-five years later, we're still using
; them for the same purpose. Some things endure.
;
; Arguments:
;   RDI = dst   - Destination address in VRAM (must be 16-byte aligned)
;   RSI = src   - Source address in RAM (should be 64-byte aligned for perf)
;   RDX = count - Bytes to copy (must be multiple of 64)
;
; Returns:
;   RAX = bytes copied (always equals count on success)
;
; Clobbers: XMM0-XMM3, RCX
; Preserves: RBX, RBP, R12-R15 (callee-saved as per SysV AMD64 ABI)
;═══════════════════════════════════════════════════════════════════════════════
ps_memcpy_to_vram:
    push    rbx                     ; Save callee-saved registers
    
    test    rdx, rdx                ; Zero-length copy?
    jz      .done                   ; Avoid division by zero and other sins
    
    mov     rcx, rdx
    shr     rcx, 6                  ; count / 64 = number of iterations
    
.loop:
    ; Load 64 bytes from RAM (likely cached)
    movdqa  xmm0, [rsi + 0]
    movdqa  xmm1, [rsi + 16]
    movdqa  xmm2, [rsi + 32]
    movdqa  xmm3, [rsi + 48]
    
    ; Store 64 bytes to VRAM (bypass cache, flow to PCIe)
    movntdq [rdi + 0],  xmm0
    movntdq [rdi + 16], xmm1
    movntdq [rdi + 32], xmm2
    movntdq [rdi + 48], xmm3
    
    add     rsi, 64
    add     rdi, 64
    dec     rcx
    jnz     .loop
    
    sfence                          ; Ensure all WC buffers are flushed
    
.done:
    mov     rax, rdx                ; Return bytes copied
    pop     rbx
    ret
```

---

## ◈ The Commit Message as Art Form

A commit message tells a story. The first line is the headline; the body is the article.

```
type(scope): short description (imperative mood, <50 chars)

Extended explanation if the change isn't self-evident. Wrap at 72 columns.
Explain *why*, not *what*—the diff shows *what*.

If fixing a bug, describe the failure mode. If optimizing, show the numbers.

Fixes: #123
Refs: #456
```

**Types**: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `style`, `chore`

### A Good Commit Message

```
fix(hmm): handle migration failure during pool exhaustion

When ps_pool_alloc() returns NULL during migrate_to_ram(), we were
returning VM_FAULT_SIGBUS, which would crash userspace with a bus error.
This is too harsh—the process might recover if we wait.

Changed to VM_FAULT_OOM, allowing the kernel's OOM retry logic to
function. The process may still be killed if memory pressure persists,
but at least the kernel decides, not us.

Reproduction: stress-ng --vm 8 --vm-bytes 90% -t 60s

Fixes: #42
```

### A Bad Commit Message

```
fixed bug
```

This tells us nothing. What bug? Where? Why? How do we know it's fixed? The git log becomes useless.

---

## ◈ The Testing Ritual

Before submitting a pull request, verify your changes don't break the universe:

```bash
# Does it compile without warnings?
make clean && make 2>&1 | grep -i warning

# Does static analysis approve?
make check

# If you have hardware, does it actually work?
sudo modprobe pseudoscopic
sudo ./tools/ps-test --all
```

### The Review Process

1. **Fork and branch** from `develop` (not `main`)
2. **Make changes** in small, atomic commits
3. **Test thoroughly** on whatever hardware you have
4. **Submit PR** against `develop`
5. **Respond to review** — We may ask for changes
6. **Celebrate** when merged (or not, HM thank you very much)

---

## ◈ Where to Focus Your Energy

### High Value, High Impact

- **Testing on diverse hardware** — V100, A100, consumer RTX, Quadro reports welcome
- **Performance profiling** — Find the 3% of code consuming 80% of time
- **Documentation** — A confused user is a lost user
- **Bug fixes in error paths** — These are where dragons live

### Medium Value

- **Consumer GPU Large BAR** — Many gamers would benefit
- **Hugepage migration (2MB/1GB)** — Reduce TLB pressure
- **NUMA optimization** — Multi-socket servers need love

### Discuss First

- **Major new features** — Must justify their complexity
- **Alternative architectures** — ARM64, RISC-V, etc.
- **Non-NVIDIA GPUs** — AMD ROCm, Intel oneAPI

---

## ◈ Communication Channels

- **Technical questions**: Open a GitHub issue
- **Security vulnerabilities**: Email `security@neuralsplines.com` (private disclosure)
- **General discussion**: GitHub Discussions (if enabled)

---

```
    ╔═══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                               ║
    ║      "Constraints breed elegance.                                             ║
    ║       The best code is the code that was never written."                      ║
    ║                                                                               ║
    ║                                        — Neural Splines Research, 2026        ║
    ║                                           Code is poetry in motion            ║
    ║                                                                               ║
    ╚═══════════════════════════════════════════════════════════════════════════════╝
```

*The repository awaits your mark.* 🍪
