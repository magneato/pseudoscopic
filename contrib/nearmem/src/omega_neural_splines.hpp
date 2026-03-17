#ifndef OMEGA_NEURAL_SPLINES_HPP
#define OMEGA_NEURAL_SPLINES_HPP
/*
 * omega_neural_splines.hpp — C++ Neural Splines™ Module
 *
 * ╔═══════════════════════════════════════════════════════════════╗
 * ║  NOTICE: This module implements Neural Splines™ technology,  ║
 * ║  a proprietary invention of Neural Splines LLC.              ║
 * ║  US Provisional Patent Application Filed.                    ║
 * ║  (c) 2026 Robert L. Sitton, Jr. — All Rights Reserved       ║
 * ╚═══════════════════════════════════════════════════════════════╝
 *
 * C++ wrapper providing RAII, type safety, and template
 * generalization over the C implementation.
 *
 * SPDX-License-Identifier: Proprietary
 */

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <array>
#include <functional>

extern "C" {
#include "omega_spline.h"
#include "omega_sitton.h"
}

namespace neural_splines {

/* ── Sitton<T>: type-safe dual number ────────────────────────── */

template<typename T>
struct Sitton {
    T real{};
    T prob{};

    constexpr Sitton() = default;
    constexpr Sitton(T r, T p = T{}) : real(r), prob(p) {}

    constexpr Sitton operator+(const Sitton &o) const
    { return {real + o.real, prob + o.prob}; }

    constexpr Sitton operator-(const Sitton &o) const
    { return {real - o.real, prob - o.prob}; }

    /* pfrak^2 = 0 */
    constexpr Sitton operator*(const Sitton &o) const
    { return {real * o.real, real * o.prob + prob * o.real}; }

    constexpr Sitton operator*(T s) const
    { return {real * s, prob * s}; }

    [[nodiscard]] constexpr T sensitivity() const
    { return (real != T{}) ? std::abs(prob / real) : std::numeric_limits<T>::infinity(); }

    [[nodiscard]] constexpr bool is_dark() const { return prob < T{}; }

    /* Convert to/from C type */
    explicit operator sitton_f32_t() const { return {static_cast<float>(real), static_cast<float>(prob)}; }
    static Sitton from_c(sitton_f32_t s) { return {static_cast<T>(s.real), static_cast<T>(s.prob)}; }
};

using Sf32 = Sitton<float>;
using Sf64 = Sitton<double>;

/* ── Spline3D: RAII wrapper ──────────────────────────────────── */

class Spline3D {
    ns_spline_3d_t spline_{};
    bool valid_{false};

public:
    Spline3D() = default;
    ~Spline3D() { if (valid_) ns_spline_3d_destroy(&spline_); }

    Spline3D(const Spline3D &) = delete;
    Spline3D &operator=(const Spline3D &) = delete;
    Spline3D(Spline3D &&o) noexcept : spline_(o.spline_), valid_(o.valid_)
    { o.valid_ = false; }

    bool init(size_t nx, size_t ny, size_t nz,
              float xn, float xx, float yn, float yx, float zn, float zx)
    {
        valid_ = (ns_spline_3d_init(&spline_, nx, ny, nz, xn, xx, yn, yx, zn, zx) == 0);
        return valid_;
    }

    [[nodiscard]] float eval(float x, float y, float z) const
    { return valid_ ? ns_spline_3d_eval(&spline_, x, y, z) : 0.0f; }

    [[nodiscard]] Sf32 eval_sitton(Sf32 x, Sf32 y, Sf32 z) const {
        if (!valid_) return {};
        auto r = ns_spline_3d_eval_sitton(&spline_,
            static_cast<sitton_f32_t>(x),
            static_cast<sitton_f32_t>(y),
            static_cast<sitton_f32_t>(z));
        return Sf32::from_c(r);
    }

    [[nodiscard]] ns_spline_3d_t *raw() { return &spline_; }
    [[nodiscard]] const ns_spline_3d_t *raw() const { return &spline_; }
};

/* ── Hierarchy: multi-resolution compression ─────────────────── */

class Hierarchy {
    ns_hierarchy_t hier_{};
    bool valid_{false};

public:
    Hierarchy() = default;
    ~Hierarchy() { if (valid_) ns_hierarchy_destroy(&hier_); }

    Hierarchy(const Hierarchy &) = delete;
    Hierarchy &operator=(const Hierarchy &) = delete;

    bool fit(const float *data, size_t w, size_t h, size_t d,
             float xn, float xx, float yn, float yx, float zn, float zx,
             float target_rms = 1e-4f, uint8_t max_levels = 4, size_t base = 8)
    {
        valid_ = (ns_hierarchy_fit(&hier_, data, w, h, d,
                  xn, xx, yn, yx, zn, zx, target_rms, max_levels, base) == 0);
        return valid_;
    }

    [[nodiscard]] float eval(float x, float y, float z) const
    { return valid_ ? ns_hierarchy_eval(&hier_, x, y, z) : 0.0f; }

    [[nodiscard]] float compression_ratio() const {
        return (hier_.compressed_bytes > 0)
            ? static_cast<float>(hier_.original_bytes) / static_cast<float>(hier_.compressed_bytes)
            : 0.0f;
    }

    [[nodiscard]] float rms_residual() const { return hier_.rms_residual; }
    void print_stats() const { if (valid_) ns_hierarchy_print_stats(&hier_); }
};

} /* namespace neural_splines */

#endif /* OMEGA_NEURAL_SPLINES_HPP */

