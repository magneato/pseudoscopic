/*
 * nearmem.hpp - C++ Wrapper for Near-Memory Computing Library
 *
 * Modern C++17 interface with:
 *   - RAII memory management
 *   - Usage hints for optimization (STREAMING, RANDOM, etc.)
 *   - Smart pointers for GPU memory
 *   - Multi-GPU support
 *   - Type-safe allocations
 *   - Exception safety
 *
 * Compiler Requirements:
 *   - C++17 or later
 *   - Same GCC version family as kernel (for ABI compatibility)
 *
 * Example:
 *   nearmem::Context ctx("/dev/psdisk0");
 *   auto buffer = ctx.alloc<float>(1024, nearmem::Usage::STREAMING);
 *   buffer[0] = 3.14f;
 *   buffer.sync_to_gpu();
 *
 * Copyright (C) 2026 Neural Splines LLC
 * License: MIT
 */

#ifndef _NEARMEM_HPP_
#define _NEARMEM_HPP_

// C++17 required
#if __cplusplus < 201703L
#error "nearmem.hpp requires C++17 or later"
#endif

#include <cstdint>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>
#include <stdexcept>
#include <functional>
#include <optional>
#include <string_view>
#include <type_traits>
#include <unordered_map>

// C API
extern "C" {
#include "nearmem.h"
#include "nearmem_gpu.h"
}

namespace nearmem {

/*
 * ════════════════════════════════════════════════════════════════════════════
 * EXCEPTIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

class Exception : public std::runtime_error {
public:
    explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
    explicit Exception(int error_code) 
        : std::runtime_error(nearmem_strerror(static_cast<nearmem_error_t>(error_code))), code_(error_code) {}
    
    int code() const noexcept { return code_; }
    
private:
    int code_ = 0;
};

class AllocationError : public Exception {
public:
    explicit AllocationError(size_t size) 
        : Exception("Failed to allocate " + std::to_string(size) + " bytes") {}
};

class DeviceError : public Exception {
public:
    explicit DeviceError(const std::string& device) 
        : Exception("Device error: " + device) {}
};

/*
 * ════════════════════════════════════════════════════════════════════════════
 * USAGE HINTS
 * ════════════════════════════════════════════════════════════════════════════
 */

/**
 * Memory usage hints for optimization
 * 
 * The allocator uses these hints to:
 *   - Choose optimal memory placement
 *   - Configure caching behavior
 *   - Pre-configure DMA settings
 *   - Optimize for access patterns
 */
enum class Usage : uint32_t {
    DEFAULT     = 0,        /**< General purpose, no specific pattern */
    
    /* Access pattern hints */
    STREAMING   = 1 << 0,   /**< Sequential access, write-once-read-once */
    RANDOM      = 1 << 1,   /**< Random access pattern */
    SEQUENTIAL  = 1 << 2,   /**< Sequential read or write */
    
    /* Lifetime hints */
    TEMPORARY   = 1 << 4,   /**< Short-lived allocation */
    PERSISTENT  = 1 << 5,   /**< Long-lived allocation */
    
    /* Access direction hints */
    READ_MOSTLY = 1 << 8,   /**< Mostly read, rarely written */
    WRITE_MOSTLY= 1 << 9,   /**< Mostly written, rarely read */
    READ_WRITE  = 1 << 10,  /**< Balanced read/write */
    
    /* GPU hints */
    GPU_ONLY    = 1 << 12,  /**< Accessed only by GPU */
    CPU_ONLY    = 1 << 13,  /**< Accessed only by CPU */
    SHARED      = 1 << 14,  /**< Shared between CPU and GPU */
    
    /* Special hints */
    TILED       = 1 << 16,  /**< Will be accessed in tiles */
    DOUBLE_BUF  = 1 << 17,  /**< Double-buffered access */
    ZERO_COPY   = 1 << 18,  /**< Minimize copies */
    
    /* Prefetch hints */
    PREFETCH_ENABLE  = 1 << 20, /**< Enable automatic prefetch */
    PREFETCH_DISABLE = 1 << 21, /**< Disable automatic prefetch */
};

inline Usage operator|(Usage a, Usage b) {
    return static_cast<Usage>(static_cast<uint32_t>(a) | static_cast<uint32_t>(b));
}

inline Usage operator&(Usage a, Usage b) {
    return static_cast<Usage>(static_cast<uint32_t>(a) & static_cast<uint32_t>(b));
}

inline bool has_flag(Usage usage, Usage flag) {
    return (static_cast<uint32_t>(usage) & static_cast<uint32_t>(flag)) != 0;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * FORWARD DECLARATIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

class Context;
template<typename T> class Buffer;
class GPUInfo;

/*
 * ════════════════════════════════════════════════════════════════════════════
 * GPU INFORMATION
 * ════════════════════════════════════════════════════════════════════════════
 */

/**
 * Information about a GPU device
 */
class GPUInfo {
public:
    int index() const { return info_.index; }
    std::string_view name() const { return info_.name; }
    std::string_view pci_address() const { return info_.pci_address; }
    
    uint64_t vram_size() const { return info_.vram_size; }
    uint64_t vram_available() const { return info_.vram_available; }
    uint64_t bar1_base() const { return info_.bar1_base; }
    uint64_t bar1_size() const { return info_.bar1_size; }
    
    std::string_view block_device() const { return info_.block_device; }
    bool is_available() const { return info_.ps_available; }
    
    bool supports_hmm() const { return info_.supports_hmm; }
    bool supports_p2p() const { return info_.supports_p2p; }
    int compute_capability() const { return info_.compute_capability; }
    
private:
    friend class Context;
    friend std::vector<GPUInfo> enumerate_gpus();
    nearmem_gpu_info_t info_{};
};

/**
 * Enumerate all available GPUs
 */
inline std::vector<GPUInfo> enumerate_gpus() {
    int count = nearmem_gpu_count();
    if (count <= 0) return {};
    
    std::vector<nearmem_gpu_info_t> raw_infos(count);
    int found = nearmem_gpu_enumerate(raw_infos.data(), count);
    
    std::vector<GPUInfo> result;
    result.reserve(found);
    
    for (int i = 0; i < found; ++i) {
        GPUInfo info;
        info.info_ = raw_infos[i];
        result.push_back(std::move(info));
    }
    
    return result;
}

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MEMORY REGION (internal)
 * ════════════════════════════════════════════════════════════════════════════
 */

namespace detail {

struct RegionDeleter {
    nearmem_ctx_t* ctx = nullptr;
    
    void operator()(nearmem_region_t* region) const {
        if (region && ctx) {
            nearmem_free(ctx, region);
            delete region;
        }
    }
};

} // namespace detail

/*
 * ════════════════════════════════════════════════════════════════════════════
 * BUFFER - Type-safe GPU memory allocation
 * ════════════════════════════════════════════════════════════════════════════
 */

/**
 * Type-safe GPU memory buffer with RAII
 * 
 * @tparam T Element type
 */
template<typename T>
class Buffer {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using iterator = T*;
    using const_iterator = const T*;
    
    Buffer() = default;
    
    Buffer(Buffer&& other) noexcept
        : ctx_(other.ctx_)
        , region_(std::move(other.region_))
        , count_(other.count_)
        , usage_(other.usage_)
    {
        other.ctx_ = nullptr;
        other.count_ = 0;
    }
    
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) {
            region_ = std::move(other.region_);
            ctx_ = other.ctx_;
            count_ = other.count_;
            usage_ = other.usage_;
            other.ctx_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }
    
    // Non-copyable (GPU memory can't be simply copied)
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    ~Buffer() = default;  // unique_ptr handles cleanup
    
    /* ═══════ Element Access ═══════ */
    
    reference operator[](size_type idx) {
        return data()[idx];
    }
    
    const_reference operator[](size_type idx) const {
        return data()[idx];
    }
    
    reference at(size_type idx) {
        if (idx >= count_) throw std::out_of_range("Buffer index out of range");
        return data()[idx];
    }
    
    const_reference at(size_type idx) const {
        if (idx >= count_) throw std::out_of_range("Buffer index out of range");
        return data()[idx];
    }
    
    pointer data() noexcept {
        return region_ ? static_cast<T*>(region_->cpu_ptr) : nullptr;
    }
    
    const_pointer data() const noexcept {
        return region_ ? static_cast<const T*>(region_->cpu_ptr) : nullptr;
    }
    
    reference front() { return data()[0]; }
    const_reference front() const { return data()[0]; }
    
    reference back() { return data()[count_ - 1]; }
    const_reference back() const { return data()[count_ - 1]; }
    
    /* ═══════ Iterators ═══════ */
    
    iterator begin() noexcept { return data(); }
    const_iterator begin() const noexcept { return data(); }
    const_iterator cbegin() const noexcept { return data(); }
    
    iterator end() noexcept { return data() + count_; }
    const_iterator end() const noexcept { return data() + count_; }
    const_iterator cend() const noexcept { return data() + count_; }
    
    /* ═══════ Capacity ═══════ */
    
    bool empty() const noexcept { return count_ == 0; }
    size_type size() const noexcept { return count_; }
    size_type size_bytes() const noexcept { return count_ * sizeof(T); }
    
    /* ═══════ GPU Operations ═══════ */
    
    /**
     * Get GPU-side pointer (for passing to CUDA kernels)
     */
    pointer gpu_data() noexcept {
        return region_ ? static_cast<T*>(region_->gpu_ptr) : nullptr;
    }
    
    /**
     * Synchronize CPU writes to GPU
     */
    void sync_to_gpu() {
        if (ctx_ && region_) {
            nearmem_sync(ctx_, NEARMEM_SYNC_CPU_TO_GPU);
        }
    }
    
    /**
     * Synchronize GPU writes to CPU
     */
    void sync_to_cpu() {
        if (ctx_ && region_) {
            nearmem_sync(ctx_, NEARMEM_SYNC_GPU_TO_CPU);
        }
    }
    
    /**
     * Get GPU VRAM offset
     */
    uint64_t gpu_offset() const noexcept {
        return region_ ? region_->offset : 0;
    }
    
    /**
     * Check if this buffer is in GPU memory
     */
    bool is_gpu_memory() const noexcept {
        return region_ && nearmem_is_gpu_memory(region_->cpu_ptr);
    }
    
    /**
     * Get usage hints
     */
    Usage usage() const noexcept { return usage_; }
    
    /**
     * Fill buffer with value
     */
    void fill(const T& value) {
        std::fill(begin(), end(), value);
    }
    
    /**
     * Copy from CPU memory
     */
    void copy_from(const T* src, size_type count) {
        if (count > count_) count = count_;
        std::copy(src, src + count, data());
    }
    
    /**
     * Copy to CPU memory
     */
    void copy_to(T* dst, size_type count) const {
        if (count > count_) count = count_;
        std::copy(data(), data() + count, dst);
    }
    
    /**
     * Check if buffer is valid
     */
    explicit operator bool() const noexcept {
        return region_ != nullptr && region_->cpu_ptr != nullptr;
    }
    
private:
    friend class Context;
    
    nearmem_ctx_t* ctx_ = nullptr;
    std::unique_ptr<nearmem_region_t, detail::RegionDeleter> region_;
    size_type count_ = 0;
    Usage usage_ = Usage::DEFAULT;
    
    // Private constructor - use Context::alloc()
    Buffer(nearmem_ctx_t* ctx, nearmem_region_t* region, size_type count, Usage usage)
        : ctx_(ctx)
        , region_(region, detail::RegionDeleter{ctx})
        , count_(count)
        , usage_(usage)
    {}
};

/*
 * ════════════════════════════════════════════════════════════════════════════
 * MEMORY MANAGER - Usage-based allocation strategies
 * ════════════════════════════════════════════════════════════════════════════
 */

/**
 * Memory manager with usage-aware allocation
 * 
 * Provides optimized allocation based on usage patterns:
 *   - STREAMING: Large sequential allocations, coalesced
 *   - RANDOM: Smaller allocations with alignment
 *   - TILED: Tile-aligned allocations
 */
class MemoryManager {
public:
    struct Stats {
        size_t total_allocated = 0;
        size_t total_freed = 0;
        size_t current_usage = 0;
        size_t peak_usage = 0;
        size_t allocation_count = 0;
        size_t free_count = 0;
    };
    
    explicit MemoryManager(nearmem_ctx_t* ctx) : ctx_(ctx) {}
    
    /**
     * Allocate memory with usage hint
     */
    void* alloc(size_t size, Usage usage = Usage::DEFAULT) {
        size_t aligned_size = align_for_usage(size, usage);
        
        auto* region = new nearmem_region_t{};
        int ret = nearmem_alloc(ctx_, region, aligned_size);
        
        if (ret != NEARMEM_OK) {
            delete region;
            return nullptr;
        }
        
        // Track allocation
        stats_.total_allocated += aligned_size;
        stats_.current_usage += aligned_size;
        stats_.allocation_count++;
        if (stats_.current_usage > stats_.peak_usage) {
            stats_.peak_usage = stats_.current_usage;
        }
        
        // Store region info for deallocation
        allocations_[region->cpu_ptr] = {region, aligned_size, usage};
        
        return region->cpu_ptr;
    }
    
    /**
     * Free previously allocated memory
     */
    void free(void* ptr) {
        auto it = allocations_.find(ptr);
        if (it == allocations_.end()) return;
        
        auto& info = it->second;
        nearmem_free(ctx_, info.region);
        delete info.region;
        
        stats_.total_freed += info.size;
        stats_.current_usage -= info.size;
        stats_.free_count++;
        
        allocations_.erase(it);
    }
    
    /**
     * Get allocation statistics
     */
    const Stats& stats() const { return stats_; }
    
    /**
     * Clear all allocations
     */
    void clear() {
        for (auto& [ptr, info] : allocations_) {
            nearmem_free(ctx_, info.region);
            delete info.region;
        }
        allocations_.clear();
        stats_.current_usage = 0;
    }
    
private:
    struct AllocInfo {
        nearmem_region_t* region;
        size_t size;
        Usage usage;
    };
    
    nearmem_ctx_t* ctx_;
    std::unordered_map<void*, AllocInfo> allocations_;
    Stats stats_{};
    
    static size_t align_for_usage(size_t size, Usage usage) {
        size_t alignment = 64;  // Default cache line
        
        if (has_flag(usage, Usage::STREAMING)) {
            alignment = 4096;  // Page aligned for DMA
        } else if (has_flag(usage, Usage::TILED)) {
            alignment = 16384; // Tile aligned (64x64 * 4)
        } else if (has_flag(usage, Usage::RANDOM)) {
            alignment = 256;   // CUDA texture alignment
        }
        
        return (size + alignment - 1) & ~(alignment - 1);
    }
};

/*
 * ════════════════════════════════════════════════════════════════════════════
 * CONTEXT - Main interface for near-memory operations
 * ════════════════════════════════════════════════════════════════════════════
 */

/**
 * Near-memory computing context
 * 
 * RAII wrapper around nearmem_ctx_t with C++ conveniences.
 */
class Context {
public:
    /**
     * Construct context for a specific device
     * @param device Path to pseudoscopic device (e.g., "/dev/psdisk0")
     * @param flags  Initialization flags
     */
    explicit Context(const std::string& device = "/dev/psdisk0", int flags = 0) {
        int ret = nearmem_init(&ctx_, device.c_str(), flags);
        if (ret != NEARMEM_OK) {
            throw DeviceError(device + ": " + nearmem_strerror(static_cast<nearmem_error_t>(ret)));
        }
        memman_ = std::make_unique<MemoryManager>(&ctx_);
    }
    
    /**
     * Construct context for a specific GPU by index
     * @param gpu_index GPU index (0-based)
     * @param flags     Initialization flags
     */
    explicit Context(int gpu_index, int flags = 0) {
        nearmem_gpu_info_t info;
        if (nearmem_gpu_get_info(gpu_index, &info) != 0) {
            throw DeviceError("GPU " + std::to_string(gpu_index) + " not found");
        }
        
        if (!info.ps_available) {
            throw DeviceError("GPU " + std::to_string(gpu_index) + 
                             " has no pseudoscopic device");
        }
        
        int ret = nearmem_init(&ctx_, info.block_device, flags);
        if (ret != NEARMEM_OK) {
            throw DeviceError(std::string(info.block_device) + ": " + 
                             nearmem_strerror(static_cast<nearmem_error_t>(ret)));
        }
        
        gpu_index_ = gpu_index;
        memman_ = std::make_unique<MemoryManager>(&ctx_);
    }
    
    ~Context() {
        memman_.reset();  // Free all allocations first
        nearmem_shutdown(&ctx_);
    }
    
    // Non-copyable
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    
    // Movable
    Context(Context&& other) noexcept
        : ctx_(other.ctx_)
        , gpu_index_(other.gpu_index_)
        , memman_(std::move(other.memman_))
    {
        other.ctx_ = {};
        other.gpu_index_ = -1;
    }
    
    Context& operator=(Context&& other) noexcept {
        if (this != &other) {
            memman_.reset();
            nearmem_shutdown(&ctx_);
            
            ctx_ = other.ctx_;
            gpu_index_ = other.gpu_index_;
            memman_ = std::move(other.memman_);
            
            other.ctx_ = {};
            other.gpu_index_ = -1;
        }
        return *this;
    }
    
    /* ═══════ Allocation ═══════ */
    
    /**
     * Allocate typed buffer
     * @tparam T Element type
     * @param count Number of elements
     * @param usage Usage hints for optimization
     */
    template<typename T>
    Buffer<T> alloc(size_t count, Usage usage = Usage::DEFAULT) {
        static_assert(std::is_trivially_copyable_v<T>,
                      "Buffer element type must be trivially copyable");
        
        size_t size = count * sizeof(T);
        auto* region = new nearmem_region_t{};
        
        int ret = nearmem_alloc(&ctx_, region, size);
        if (ret != NEARMEM_OK) {
            delete region;
            throw AllocationError(size);
        }
        
        return Buffer<T>(&ctx_, region, count, usage);
    }
    
    /**
     * Allocate raw memory with usage hint
     */
    void* alloc_raw(size_t size, Usage usage = Usage::DEFAULT) {
        return memman_->alloc(size, usage);
    }
    
    /**
     * Free raw memory
     */
    void free_raw(void* ptr) {
        memman_->free(ptr);
    }
    
    /* ═══════ Synchronization ═══════ */
    
    /**
     * Sync CPU writes to GPU
     */
    void sync_to_gpu() {
        nearmem_sync(&ctx_, NEARMEM_SYNC_CPU_TO_GPU);
    }
    
    /**
     * Sync GPU writes to CPU
     */
    void sync_to_cpu() {
        nearmem_sync(&ctx_, NEARMEM_SYNC_GPU_TO_CPU);
    }
    
    /**
     * Full bidirectional sync
     */
    void sync() {
        sync_to_gpu();
        sync_to_cpu();
    }
    
    /* ═══════ Device Information ═══════ */
    
    /**
     * Get VRAM size in bytes
     */
    size_t vram_size() const { return ctx_.ps_size; }
    
    /**
     * Get GPU index (-1 if unknown)
     */
    int gpu_index() const { return gpu_index_; }
    
    /**
     * Get memory manager statistics
     */
    const MemoryManager::Stats& memory_stats() const {
        return memman_->stats();
    }
    
    /**
     * Get raw C context (for interop)
     */
    nearmem_ctx_t* raw() { return &ctx_; }
    const nearmem_ctx_t* raw() const { return &ctx_; }
    
    /**
     * Check if context is valid
     */
    explicit operator bool() const {
        return ctx_.ps_size > 0;
    }
    
private:
    nearmem_ctx_t ctx_{};
    int gpu_index_ = -1;
    std::unique_ptr<MemoryManager> memman_;
};

/*
 * ════════════════════════════════════════════════════════════════════════════
 * CONVENIENCE FUNCTIONS
 * ════════════════════════════════════════════════════════════════════════════
 */

/**
 * Get number of available GPUs
 */
inline int gpu_count() {
    return nearmem_gpu_count();
}

/**
 * Check if a pointer is in GPU memory
 */
inline bool is_gpu_memory(const void* ptr) {
    return nearmem_is_gpu_memory(ptr);
}

/**
 * Get GPU index for a pointer
 */
inline int get_gpu_for_ptr(const void* ptr) {
    return nearmem_get_gpu_for_ptr(ptr);
}

/**
 * Get VRAM base address for a GPU
 */
inline uint64_t vram_base(int gpu_index) {
    return nearmem_gpu_get_vram_base(gpu_index);
}

/**
 * Get VRAM size for a GPU
 */
inline uint64_t vram_size(int gpu_index) {
    return nearmem_gpu_get_vram_size(gpu_index);
}

} // namespace nearmem

/*
 * ════════════════════════════════════════════════════════════════════════════
 * COMPILER VERSION CHECK
 * ════════════════════════════════════════════════════════════════════════════
 *
 * The kernel module must be compiled with the same GCC version family as
 * the kernel itself. This C++ wrapper should also use a compatible version
 * to ensure ABI stability.
 *
 * Check at compile time:
 */

#if defined(__GNUC__) && !defined(__clang__)
    // GCC version as single number: major * 10000 + minor * 100 + patch
    #define NEARMEM_GCC_VERSION (__GNUC__ * 10000 + __GNUC_MINOR__ * 100 + __GNUC_PATCHLEVEL__)
    
    // Minimum supported: GCC 7 for C++17
    #if NEARMEM_GCC_VERSION < 70000
        #error "nearmem.hpp requires GCC 7.0 or later for C++17 support"
    #endif
    
    // Warn if using very new GCC (might have ABI changes)
    #if NEARMEM_GCC_VERSION >= 140000
        #pragma message "Note: Using GCC 14+. Ensure kernel was compiled with compatible version."
    #endif
#endif

#endif /* _NEARMEM_HPP_ */
