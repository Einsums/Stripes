//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// rsqrt, erf and erfc (Math.hpp) against references accurate well past the result's precision: for
// double, long double erfl, erfcl and 1 / sqrtl where long double is wider than double, and a
// double-double evaluation (below) where it is not; for float, the double functions. The worst errors measured when they were
// written, over each function's whole range with subnormal results included:
//
//   rsqrt   1.49 ulp
//   erf     0.75 ulp below |x| = 0.5, 1.8 ulp above
//   erfc    1.1 ulp below x = 0.5, 4.0 ulp above (3.8 for float)
//
// and the bounds checked here are 1.6, 2.0 and 4.5 ulp. Special values are checked exactly. That the
// scalar instantiations match the vector lanes bit for bit is checked in GenericKernel.

#include <Stripes/Math.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <ostream>
#include <type_traits>

#include <catch2/catch_all.hpp>

namespace simd = stripes;

namespace {

/// Points per sweep, divided by STRIPES_TEST_SWEEP_DIVISOR where it is set: CI sets it under QEMU,
/// where the full sweeps (and, on aarch64 Linux, long double's software arithmetic) take minutes.
/// The native jobs run them in full.
long sweep(long n) {
    static long const divisor = [] {
        char const *s = std::getenv("STRIPES_TEST_SWEEP_DIVISOR");
        long const  d = s != nullptr ? std::atol(s) : 1;
        return d > 1 ? d : 1L;
    }();
    return std::max(n / divisor, 64L);
}

template <typename T>
double ulp_error(T got, long double ref) {
    if (std::isnan(got) || std::isinf(got)) {
        return 1e9;
    }
    T const r   = static_cast<T>(ref);
    T       ulp = std::nextafter(std::fabs(r), std::numeric_limits<T>::infinity()) - std::fabs(r);
    if (r == T(0)) {
        ulp = std::numeric_limits<T>::denorm_min();
    }
    return static_cast<double>(std::fabs(static_cast<long double>(got) - ref) / static_cast<long double>(ulp));
}

// ---------------------------------------------------------------------------------------------------
// A double-double reference for erf and erfc, for platforms whose long double is double (macOS on
// arm64, Windows). Their own erf and erfc are 2.5 to 8 ulp off in places, as their CI runs showed,
// so they cannot judge a 2 ulp bound; these are good to about 100 bits. Where long double is wider,
// the case "the double-double reference agrees with erfl and erfcl" checks them against it.
// ---------------------------------------------------------------------------------------------------

/// VecErfDoubleDouble_test sets STRIPES_TEST_DOUBLE_DOUBLE_REFERENCE, so the platforms with a wide
/// long double also run the accuracy cases against the reference the others depend on.
#if defined(STRIPES_TEST_DOUBLE_DOUBLE_REFERENCE)
constexpr bool wide_long_double = false;
#else
constexpr bool wide_long_double = std::numeric_limits<long double>::digits >= 64;
#endif

/// hi + lo, with |lo| at most half an ulp of hi.
struct DD {
    double hi = 0, lo = 0;
};

DD quick_two_sum(double a, double b) {
    double const s = a + b;
    return {s, b - (s - a)};
}
DD two_sum(double a, double b) {
    double const s  = a + b;
    double const bb = s - a;
    return {s, (a - (s - bb)) + (b - bb)};
}
DD two_prod(double a, double b) {
    double const p = a * b;
    return {p, std::fma(a, b, -p)};
}
DD operator-(DD x) {
    return {-x.hi, -x.lo};
}
DD operator+(DD x, DD y) {
    DD const s = two_sum(x.hi, y.hi);
    DD const t = two_sum(x.lo, y.lo);
    DD const u = quick_two_sum(s.hi, s.lo + t.hi);
    return quick_two_sum(u.hi, u.lo + t.lo);
}
DD operator-(DD x, DD y) {
    return x + -y;
}
DD operator*(DD x, DD y) {
    DD const p = two_prod(x.hi, y.hi);
    return quick_two_sum(p.hi, p.lo + (x.hi * y.lo + x.lo * y.hi));
}
DD operator/(DD x, DD y) {
    double const q1 = x.hi / y.hi;
    DD const     r1 = x - y * DD{q1};
    double const q2 = r1.hi / y.hi;
    DD const     r2 = r1 - y * DD{q2};
    double const q3 = r2.hi / y.hi;
    return quick_two_sum(q1, q2) + DD{q3};
}
DD ldexp(DD x, int e) {
    return {std::ldexp(x.hi, e), std::ldexp(x.lo, e)};
}

constexpr DD two_over_sqrt_pi{0x1.20dd750429b6dp+0, 0x1.1ae3a914fed80p-56};
constexpr DD one_over_sqrt_pi{0x1.20dd750429b6dp-1, 0x1.1ae3a914fed80p-57};
constexpr DD ln2{0x1.62e42fefa39efp-1, 0x1.abc9e3b39803fp-56};

/// value * 2^scale: erfc reaches the subnormals, where lo alone would underflow.
struct Exact {
    DD  value;
    int scale = 0;
};

std::ostream &operator<<(std::ostream &os, Exact const &e) {
    return os << std::setprecision(17) << e.value.hi << " + " << e.value.lo << " (times 2^" << e.scale << ")";
}

/// erf(x) for |x| <= 2 by its Taylor series, 2 / sqrt(pi) sum (-1)^n x^(2n + 1) / (n! (2n + 1)). The
/// largest term at |x| = 2 is about e^4, so cancellation costs some 6 of the 106 bits.
DD erf_series(double x) {
    DD const x2   = two_prod(x, x);
    DD       term = {x};
    DD       sum  = {x};
    for (int n = 1; n < 200; ++n) {
        term          = -(term * x2) / DD{static_cast<double>(n)};
        DD const part = term / DD{static_cast<double>(2 * n + 1)};
        sum           = sum + part;
        if (std::fabs(part.hi) < 1e-36 * std::fabs(sum.hi)) {
            break;
        }
    }
    return sum * two_over_sqrt_pi;
}

/// e^(-x^2) as value * 2^scale: n ln 2 taken out, the rest divided by 2^9, a Taylor series, and nine
/// squarings.
Exact exp_minus_square(double x) {
    DD const     s = two_prod(x, x);
    double const k = std::nearbyint(s.hi / ln2.hi);
    DD const     r = ldexp(ln2 * DD{k} - s, -9); // -x^2 + k ln 2, over 512: |r| < 7e-4
    DD           e = {1.0};
    DD           t = {1.0};
    for (int n = 1; n <= 14; ++n) {
        t = t * r / DD{static_cast<double>(n)};
        e = e + t;
    }
    for (int i = 0; i < 9; ++i) {
        e = e * e;
    }
    return {e, -static_cast<int>(k)};
}

/// erfc(x) for x >= 2 by its continued fraction, e^(-x^2) / (sqrt(pi) (x + (1/2) / (x + 1 / (x + (3/2)
/// / (x + ...))))), evaluated from the back; the term count gives well over 100 bits from x = 2 up.
Exact erfc_fraction(double x) {
    int const n = 40 + static_cast<int>(800.0 / (x * x));
    DD        f = {x};
    for (int k = n; k >= 1; --k) {
        f = DD{x} + DD{0.5 * k} / f;
    }
    Exact e = exp_minus_square(x);
    e.value = e.value * one_over_sqrt_pi / f;
    return e;
}

Exact dd_erf(double x) {
    double const a = std::fabs(x);
    if (a <= 2) {
        return {erf_series(x)};
    }
    Exact const c = erfc_fraction(a);
    DD const    v = DD{1.0} - ldexp(c.value, c.scale);
    return {x < 0 ? -v : v};
}

Exact dd_erfc(double x) {
    if (std::fabs(x) <= 2) {
        return {DD{1.0} - erf_series(x)};
    }
    if (x > 0) {
        return erfc_fraction(x);
    }
    Exact const c = erfc_fraction(-x);
    return {DD{2.0} - ldexp(c.value, c.scale)};
}

/// Error of got, in ulp of the result, against value * 2^scale.
template <typename T>
double ulp_error(T got, Exact ref) {
    if (std::isnan(got) || std::isinf(got)) {
        return 1e9;
    }
    T const r   = static_cast<T>(std::ldexp(ref.value.hi, ref.scale) + std::ldexp(ref.value.lo, ref.scale));
    T       ulp = std::nextafter(std::fabs(r), std::numeric_limits<T>::infinity()) - std::fabs(r);
    if (r == T(0)) {
        ulp = std::numeric_limits<T>::denorm_min();
    }
    // At the reference's own scale, where got (a double at most 2) and the difference are exact.
    double const g = std::ldexp(static_cast<double>(got), -ref.scale);
    return std::fabs((g - ref.value.hi) - ref.value.lo) / std::ldexp(static_cast<double>(ulp), -ref.scale);
}

template <typename T>
auto ref_erf(T x) {
    if constexpr (!std::is_same_v<T, double>) {
        return static_cast<long double>(std::erf(static_cast<double>(x)));
    } else if constexpr (wide_long_double) {
        return erfl(static_cast<long double>(x));
    } else {
        return dd_erf(x);
    }
}
template <typename T>
auto ref_erfc(T x) {
    if constexpr (!std::is_same_v<T, double>) {
        return static_cast<long double>(std::erfc(static_cast<double>(x)));
    } else if constexpr (wide_long_double) {
        return erfcl(static_cast<long double>(x));
    } else {
        return dd_erfc(x);
    }
}

/// Checks f against ref at n points evenly spaced in [lo, hi] (log-spaced when geometric is set),
/// run as vectors of V, to within bound ulp.
template <typename V, typename F, typename Ref>
void check_range(simd::scalar_t<V> lo, simd::scalar_t<V> hi, long n, F f, Ref ref, double bound, bool geometric = false) {
    using T         = simd::scalar_t<V>;
    constexpr int L = simd::lanes_v<V>;
    T             x[L], y[L];
    double        worst = 0;
    n                   = sweep(n);
    for (long i = 0; i < n; i += L) {
        for (int j = 0; j < L; ++j) {
            double const t = static_cast<double>(i + j) / static_cast<double>(n);
            x[j]           = geometric ? static_cast<T>(static_cast<double>(lo) * std::pow(static_cast<double>(hi / lo), t))
                                       : lo + (hi - lo) * static_cast<T>(t);
        }
        simd::store(y, f(simd::load<V>(x)));
        for (int j = 0; j < L; ++j) {
            double const e = ulp_error<T>(y[j], ref(x[j]));
            if (e > worst) {
                worst = e;
                // In full, so a failure says whether the result or the reference is off: where long
                // double is no wider than double (macOS on arm64, Windows) the reference is the
                // platform's own double erf.
                INFO(std::setprecision(17) << "x = " << x[j] << ", got " << y[j] << ", reference " << std::setprecision(21) << ref(x[j])
                                           << " (long double has " << std::numeric_limits<long double>::digits << " bits)");
                CHECK(e <= bound);
            }
        }
    }
}

template <typename T, typename F>
T lane0(F f, T x) {
    T out[simd::lanes<T>];
    simd::store(out, f(simd::splat<simd::Vec<T>>(x)));
    return out[0];
}

} // namespace

TEMPLATE_TEST_CASE("rsqrt is within 1.6 ulp of 1 / sqrt", "[simd][math][rsqrt]", float, double) {
    using T       = TestType;
    using lim     = std::numeric_limits<T>;
    auto const f  = [](auto v) { return simd::rsqrt(v); };
    T const    lo = std::is_same_v<T, double> ? T(1e-300) : T(1e-30);
    check_range<simd::Vec<T>>(lo, T(1) / lo, 1000000, f, [](T x) { return 1.0L / sqrtl(static_cast<long double>(x)); }, 1.6, true);
    CHECK(lane0<T>(f, T(4)) == T(0.5));
    CHECK(lane0<T>(f, T(0)) == lim::infinity());
    CHECK(lane0<T>(f, T(-0.0)) == -lim::infinity());
    CHECK(lane0<T>(f, lim::infinity()) == T(0));
    CHECK(std::isnan(lane0<T>(f, T(-1))));
    CHECK(std::isnan(lane0<T>(f, lim::quiet_NaN())));
}

TEMPLATE_TEST_CASE("erf is within 2 ulp of a correctly rounded erf", "[simd][math][erf]", float, double) {
    using T      = TestType;
    using lim    = std::numeric_limits<T>;
    using Wide   = simd::Vec<T, 2 * simd::lanes<float>>;
    auto const f = [](auto v) { return simd::erf(v); };
    auto const r = [](T x) { return ref_erf(x); };
    check_range<simd::Vec<T>>(T(-6), T(6), 1000000, f, r, 2.0);
    check_range<simd::Vec<T>>(T(-0.5), T(0.5), 400000, f, r, 2.0);
    check_range<simd::Vec<T>>(T(1e-30), T(0.5), 200000, f, r, 2.0, true);
    check_range<Wide>(T(-6), T(6), 200000, f, r, 2.0);
    CHECK(lane0<T>(f, T(0)) == T(0));
    CHECK(std::signbit(lane0<T>(f, T(-0.0))));
    CHECK(lane0<T>(f, lim::infinity()) == T(1));
    CHECK(lane0<T>(f, -lim::infinity()) == T(-1));
    CHECK(lane0<T>(f, T(100)) == T(1));
    CHECK(std::isnan(lane0<T>(f, lim::quiet_NaN())));
    // Odd, exactly.
    for (T x : {T(0.1), T(0.49), T(0.5), T(1.3), T(3.7)}) {
        CHECK(lane0<T>(f, -x) == -lane0<T>(f, x));
    }
}

TEMPLATE_TEST_CASE("erfc is within 4.5 ulp of a correctly rounded erfc", "[simd][math][erfc]", float, double) {
    using T        = TestType;
    using lim      = std::numeric_limits<T>;
    using Wide     = simd::Vec<T, 2 * simd::lanes<float>>;
    auto const f   = [](auto v) { return simd::erfc(v); };
    auto const r   = [](T x) { return ref_erfc(x); };
    T const    top = std::is_same_v<T, double> ? T(27.25) : T(10.05); // past here erfc is below the smallest subnormal
    check_range<simd::Vec<T>>(T(-6), T(0.5), 400000, f, r, 4.5);
    check_range<simd::Vec<T>>(T(0.5), top, 2000000, f, r, 4.5);
    check_range<Wide>(T(-6), top, 200000, f, r, 4.5);
    CHECK(lane0<T>(f, T(0)) == T(1));
    CHECK(lane0<T>(f, lim::infinity()) == T(0));
    CHECK(lane0<T>(f, -lim::infinity()) == T(2));
    CHECK(lane0<T>(f, T(100)) == T(0));
    CHECK(lane0<T>(f, T(-100)) == T(2));
    CHECK(std::isnan(lane0<T>(f, lim::quiet_NaN())));
    // Subnormal results are reached, not flushed.
    T const tiny = std::is_same_v<T, double> ? T(26.8) : T(9.6);
    CHECK(lane0<T>(f, tiny) > T(0));
    CHECK(lane0<T>(f, tiny) < lim::min());
}

TEST_CASE("the double-double reference agrees with erfl and erfcl", "[simd][math][erf][reference]") {
    // Where long double is wider than double, erfl and erfcl judge the double-double reference the
    // other platforms use, compared in long double: its error must be a small fraction of an ulp of
    // double everywhere the tests evaluate it. (x87 erfl is itself good to about 2^-11 ulp of double.)
    if constexpr (!wide_long_double) {
        SKIP("long double is double here, so there is nothing more accurate to check the reference against");
    } else {
        auto const gap = [](long double want, Exact got) {
            long double const value = std::ldexp(static_cast<long double>(got.value.hi), got.scale) +
                                      std::ldexp(static_cast<long double>(got.value.lo), got.scale);
            double const      r     = static_cast<double>(want);
            double            ulp   = std::nextafter(std::fabs(r), std::numeric_limits<double>::infinity()) - std::fabs(r);
            if (r == 0) {
                ulp = std::numeric_limits<double>::denorm_min();
            }
            return static_cast<double>(std::fabs(value - want) / static_cast<long double>(ulp));
        };
        double     worst_erf = 0, worst_erfc = 0;
        long const n_erf = sweep(120000);
        for (long i = 0; i <= n_erf; ++i) {
            double const x = -6.0 + 12.0 * static_cast<double>(i) / static_cast<double>(n_erf);
            worst_erf      = std::max(worst_erf, gap(erfl(x), dd_erf(x)));
        }
        long const n_erfc = sweep(200000);
        for (long i = 0; i <= n_erfc; ++i) {
            double const x = -6.0 + 33.25 * static_cast<double>(i) / static_cast<double>(n_erfc);
            worst_erfc     = std::max(worst_erfc, gap(erfcl(x), dd_erfc(x)));
        }
        INFO("worst erf " << worst_erf << " ulp, worst erfc " << worst_erfc << " ulp of double");
        CHECK(worst_erf <= 0.01);
        CHECK(worst_erfc <= 0.01);
    }
}
