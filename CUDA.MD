# Faster than CUDA?

*The title is clickbait. I apologize. Sort of.*

---

## The Audacity of Asking the Question

Every time I post about **Pseudoscopic**—our open-source project that treats GPU VRAM as directly-addressable system memory—someone in the comments throws down the CUDA gauntlet:

> "But CUDA exists. Why wouldn't you just use CUDA?"

Ah, yes. CUDA. The hammer so successful it convinced an entire generation that every compute problem is a nail. The framework so beloved that NVIDIA can charge whatever they want for H100s and we smile through our tears.

Listen, I use CUDA. I *like* CUDA. But here's the thing nobody at NVIDIA's marketing department mentions in the keynotes:

**CUDA's dirty secret is that most of its time isn't spent computing. It's spent copying.**

---

## A Tale of Two Pipelines

Picture this: You're an ML engineer at $COMPANY (congratulations on your RSUs). You need to process 50GB of logs looking for "ERROR" patterns. Classic grep problem, but at scale. You think: "I have a Tesla P40 with 24GB of GDDR5X sitting idle! Let me throw CUDA at this."

Here's what happens in Traditional CUDA Land™:

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: cudaMemcpy (Host → Device)                             │
│          50 GB ÷ 12.4 GB/s (PCIe Gen3) ≈ 4.0 seconds            │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: kernel<<<launch>>>                                     │
│          GPU pattern matching at 700 GB/s ≈ 0.2 seconds         │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: cudaMemcpy (Device → Host)                             │
│          Results transfer ≈ 0.01 seconds                        │
├─────────────────────────────────────────────────────────────────┤
│  TOTAL: 4.21 seconds                                            │
│  Time GPU actually computed: 4.7%                               │
└─────────────────────────────────────────────────────────────────┘
```

Four seconds of romantic PCIe bandwidth negotiation. Two hundred milliseconds of actual work. And somewhere, Jensen Huang sheds a single leather-clad tear because you're using his $10,000 compute monster as a glorified DMA controller.

---

## "But What About Unified Memory?"

I see you reaching for `cudaMallocManaged()`. Noble instinct. Wrong target.

Unified Memory is NVIDIA's answer to the question: "What if we made the programmer's life easier by hiding all the complexity... and then charging them for it at runtime?"

It works. Until it doesn't. Ever watched your carefully-optimized neural network suddenly take 10x longer because the CUDA driver decided to page-fault its way through your tensors? The runtime gods giveth, and the runtime gods taketh away—usually in production, usually at 3 AM, usually on the demo before the investor meeting.

Unified Memory is training wheels. Very expensive training wheels that occasionally lock up.

---

## The Pseudoscopic Heresy

What if I told you there's another way? What if the data just... *stayed where it is*?

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: Data is already in VRAM                                │
│          (You wrote it there via mmap/BAR1)                     │
│          Transfer time: 0 seconds                               │
├─────────────────────────────────────────────────────────────────┤
│  Step 2: GPU compute in-place                                   │
│          Pattern matching at 700 GB/s ≈ 0.2 seconds             │
├─────────────────────────────────────────────────────────────────┤
│  Step 3: CPU reads results (via mmap)                           │
│          Just the summary, not 50GB ≈ 0 seconds                 │
├─────────────────────────────────────────────────────────────────┤
│  TOTAL: 0.2 seconds                                             │
│  Time GPU actually computed: 100%                               │
└─────────────────────────────────────────────────────────────────┘
```

**22x faster.** Not through smarter algorithms. Not through tensor cores. Through the radical innovation of *not moving data you don't need to move*.

I can already hear the CUDA purists typing furiously. Their comments will read like Medium posts about hustle culture: technically correct, spiritually empty.

"*But BAR1 access is slower than device memory!*"

Yes. At the byte level. *I don't care about byte-level access.* I care about the entire workflow. If my choice is 100ns PCIe round-trip latency versus 4 seconds of bulk transfer purgatory, I'll take the latency every time.

"*This only works for specific workloads!*"

Yes! Correct! Unlike CUDA, which famously works for all workloads, including that guy who tried to train a ResNet on CSV parsing. Everything has a domain. The question is whether you're in it.

---

## What Pseudoscopic Actually Does

Let me be precise, because precision matters and LinkedIn is tired of vague hype-posts:

**Pseudoscopic** is a Linux kernel module that exposes GPU VRAM as a block device. When you `mmap()` `/dev/psdisk0`, you get a CPU pointer that writes directly to VRAM through the PCIe BAR1 aperture. No copies. No kernel buffers. Just CPU ↔ GPU address space unification.

Here's the trick that makes it sing: **the same physical memory is accessible by CUDA**.

```c
nearmem_region_t region;
nearmem_alloc(&ctx, &region, 16 * GB);

// CPU writes directly to VRAM
memcpy(region.cpu_ptr, my_fifty_gigs, 50 * GB);

// GPU computes on the same bytes
nearmem_histogram(&ctx, &region, byte_counts);

// CPU reads the 2KB result
printf("Errors: %lu\n", byte_counts['E']);
```

Same physical memory. Two access modalities. No `cudaMemcpy`. The GPU becomes a *memory-side accelerator*—a coprocessor that transforms data in-place while the CPU orchestrates.

---

## When This Wins (And When It Doesn't)

I'm not here to sell snake oil. This approach isn't for everyone. Here's the honest breakdown:

### Use Near-Memory When:

- **Data is memory-bound, not compute-bound** — Log analysis, grep, validation
- **You're processing streaming data** — Ingest → process → repeat
- **Transfer time dominates your pipeline** — Most "big data" workloads
- **The data is already on the GPU** — Inference buffers, VRAM caches
- **You want a blazing-fast ramdisk** — Temporary storage at 700 GB/s internal

### Stick With Traditional CUDA When:

- **You need iterative CPU↔GPU feedback** — Training loops, interactive simulations
- **Your dataset fits in shared memory** — The kernel is the bottleneck, not transfer
- **Random access patterns dominate** — PCIe latency will eat you alive
- **You need cuDNN/cuBLAS** — Battle-tested libraries don't care about your clever architecture

---

## The Philosophy Behind the Hack

Here's where I get pretentious. Bear with me.

The CUDA programming model assumes a fundamental asymmetry: CPU is orchestrator, GPU is compute oracle. You send data *to* the oracle. You receive answers *from* the oracle. This is the church of GPGPU, and for twenty years, it's been correct.

But paradigms calcify. The assumption that "GPU memory is device memory" is an artifact of *hardware that shipped in 2006*. Modern GPUs have Large BAR—they expose their entire VRAM through PCIe. Modern CPUs have `mmap()`. Modern kernels have heterogeneous memory management.

**The impedance mismatch between CPU and GPU memory isn't fundamental. It's legacy.**

Pseudoscopic doesn't fight CUDA. It doesn't replace CUDA. It *sidesteps* the cudaMemcpy tax by refusing to pay it. When your workload is "transform bytes in place," you don't need to round-trip through cudaMemcpy. You need a CPU pointer and a GPU kernel that share a physical page.

That's all this is. One heretical insight, followed by 2,500 lines of kernel module to make it real.

---

## The Numbers That Made Me Write This Post

I ran benchmarks. You can too. The code is MIT-licensed and sitting on GitHub.

| Workload | Traditional CUDA | Near-Memory | Speedup |
|----------|------------------|-------------|---------|
| 50GB log grep | 4.4s | 0.2s | **22x** |
| 16GB histogram | 1.5s | 0.07s | **21x** |
| 24GB byte transform | 2.2s | 0.1s | **22x** |
| 8GB radix sort | 0.8s | 0.3s | **2.7x** |

The sort is interesting—it shows that when compute time approaches transfer time, the gains shrink. This isn't magic. It's just math.

---

## What I'm Really Saying

CUDA is phenomenal. It unlocked a generation of computing. It made neural networks economically viable. Jensen Huang deserves every billion.

But tools have contexts. Hammers have use cases. When the CUDA programming model force-marches your data through a PCIe toll booth just to apply a transformation that should happen in-place, maybe the tool isn't the problem. Maybe the mental model is.

**Faster than CUDA?** In some contexts, yes. Dramatically. Not because we're smarter than NVIDIA's army of PhDs, but because *not moving data* is always faster than *moving data*.

The secret isn't better algorithms. It's realizing which algorithms you don't have to run at all.

---

## Try It Yourself

The project is open source:

🌐 [pseudoscopic.ai](https://pseudoscopic.ai)  
💻 [github.com/magneato/pseudoscopic](https://github.com/magneato/pseudoscopic)

Load the module. Mount your VRAM. Process your logs. Then come back and tell me I'm wrong.

Or don't. Those four seconds of `cudaMemcpy` are a great time to check Slack.

---

*Neural Splines Research, 2025*  
*Asymmetric solutions.*

---

**#CUDA #GPU #Systems #Linux #HighPerformanceComputing #Startups #OpenSource #Engineering**

---

But what about AMD? Time to mRock on!

