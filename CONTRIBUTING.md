# Contributing to Pseudoscopic

Thank you for considering contributing to Pseudoscopic! This project embodies a philosophy of minimal elegance—every line of code should earn its place.

## Philosophy

Before contributing, understand what we're building:

1. **Minimal surface area**: We don't add features for the sake of features
2. **Bulletproof reliability**: Every error path must be handled
3. **Performance where it matters**: Assembly for hot paths, clarity elsewhere
4. **Beauty in code**: If it's not readable, it's not maintainable

## Code of Conduct

Be excellent to each other. This is a technical project; keep discussions technical. We evaluate code, not people.

## Getting Started

### Development Environment

```bash
# Clone the repository
git clone https://github.com/magneato/pseudoscopic.git
cd pseudoscopic

# Install development dependencies (Ubuntu)
sudo apt install build-essential linux-headers-$(uname -r) nasm \
    sparse cppcheck

# Build with debug symbols
make DEBUG=1

# Run static analysis
make check
```

### Project Structure

```
pseudoscopic/
├── src/
│   ├── core/          # Module entry, PCI handling, pool management
│   ├── hmm/           # HMM integration, migration, notifiers
│   ├── dma/           # DMA engine interface
│   └── asm/           # Hand-optimized assembly (NASM)
├── include/           # Public headers
├── tools/             # Userspace utilities
├── scripts/           # Installation and helper scripts
└── .github/           # CI workflows
```

## Making Changes

### Before You Start

1. **Check existing issues** - Someone may already be working on it
2. **Open an issue first** - For significant changes, discuss before coding
3. **One thing at a time** - Keep PRs focused and reviewable

### Coding Standards

#### C Code

Follow the [Linux kernel coding style](https://www.kernel.org/doc/html/latest/process/coding-style.html):

```c
/* Good: Clear, single-purpose function */
static int ps_pool_alloc_page(struct ps_pool *pool, struct page **page_out)
{
    struct page *page;
    unsigned long flags;
    
    spin_lock_irqsave(&pool->lock, flags);
    
    page = pool->free_list;
    if (!page) {
        spin_unlock_irqrestore(&pool->lock, flags);
        return -ENOMEM;
    }
    
    pool->free_list = page->zone_device_data;
    atomic_long_dec(&pool->free);
    
    spin_unlock_irqrestore(&pool->lock, flags);
    
    *page_out = page;
    return 0;
}
```

**Key points:**
- Tabs for indentation (8 spaces width)
- 80 column limit (soft) 
- Braces on same line for functions, after for control flow
- Check every return value
- Clean up on error paths

#### Assembly (NASM)

```nasm
; Good: Clear documentation, structured code
;-----------------------------------------------------------------------------
; ps_memcpy_to_vram - Copy from system RAM to VRAM
;
; Arguments:
;   rdi = dst   - Destination in VRAM
;   rsi = src   - Source in system RAM  
;   rdx = count - Bytes to copy (must be multiple of 64)
;
; Preserves: rbx, rbp, r12-r15
;-----------------------------------------------------------------------------
ps_memcpy_to_vram:
    push    rbx
    
    test    rdx, rdx
    jz      .done
    
.loop:
    ; Process 64 bytes per iteration
    movdqa  xmm0, [rsi]
    movntdq [rdi], xmm0
    ; ...
```

**Key points:**
- Document calling convention
- Preserve callee-saved registers
- Handle edge cases (zero length, etc.)
- Comment non-obvious operations

### Commit Messages

Follow conventional commits:

```
type(scope): short description

Longer explanation if needed. Wrap at 72 columns.

Fixes: #123
```

Types: `feat`, `fix`, `perf`, `docs`, `test`, `refactor`, `style`

**Good:**
```
fix(hmm): handle migration failure during pool exhaustion

When ps_pool_alloc() returns NULL during migrate_to_ram(), we were
returning VM_FAULT_SIGBUS which could crash userspace. Changed to
VM_FAULT_OOM to allow kernel retry logic to function.

Fixes: #42
```

**Bad:**
```
fixed bug
```

### Testing

Before submitting:

```bash
# Build cleanly
make clean && make

# Run static analysis
make check

# Test on actual hardware if possible
sudo modprobe pseudoscopic
sudo ./tools/ps-test --all
```

### Pull Request Process

1. **Fork and branch** - Create a feature branch from `develop`
2. **Make changes** - Keep commits atomic and well-described
3. **Test thoroughly** - Both build and runtime
4. **Submit PR** - Against `develop`, not `main`
5. **Respond to review** - We may ask for changes
6. **Squash if requested** - Keep history clean

## Areas for Contribution

### High Priority

- **Testing on more hardware** - V100, A100, Quadro RTX reports welcome
- **Performance optimization** - Profiling and improving hot paths
- **Documentation** - Improving clarity and adding examples
- **Bug fixes** - Especially in error handling paths

### Medium Priority

- **Consumer GPU support** - Resizable BAR implementation
- **Hugepage support** - 2MB page migration
- **NUMA awareness** - Multi-socket optimization

### Low Priority (Discuss First)

- **New features** - Must justify complexity
- **Alternative architectures** - ARM64, etc.
- **Non-NVIDIA GPUs** - AMD, Intel

## Questions?

- **Technical questions**: Open a GitHub issue
- **Security issues**: Email security@neuralsplines.com (do not open public issues)

---

*"Constraints breed elegance."*

— Neural Splines Research
