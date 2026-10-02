//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// The scalar half of Stripes in a CUDA translation unit. One body, written over its value type as a
// kernel generator writes it, runs on the device one problem per thread and in this file's host code
// one problem per loop iteration; every result must match bit for bit. The CPU suites (GenericKernel,
// VecExp, VecErf) already hold the scalar forms to one vector lane each, so this closes the chain
// from a Vec<T> lane to a GPU thread.
//
// Exits 0 when every result matches, 1 on a mismatch or a CUDA error, and 77 (ctest's Skipped) when
// the machine has no CUDA device: the build alone then shows the headers compile for the device.

#include <Stripes/Generic.hpp>
#include <Stripes/Math.hpp>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <vector>

namespace simd = stripes;

namespace {

/// The operations a generated kernel calls, each written once over V. Writes result k of problem i
/// at out[k * n + i].
constexpr int results = 25;

template <typename V>
__host__ __device__ void operations(V x, V y, simd::scalar_t<V> const *table, int table_size, simd::scalar_t<V> *out, int n, int i) {
    using S = simd::scalar_t<V>;
    using I = simd::gather_index_t<S>;

    // An index into the table from x: floor(|x|), held below the table's end. min returns its second
    // argument for a NaN, so a NaN lands on the last entry.
    V const    clamped = simd::min(simd::floor(simd::abs(x)), simd::splat<V>(S(table_size - 1)));
    auto const idx     = simd::convert<I>(clamped);

    V t = simd::fmadd(x, y, simd::splat<V>(S(0.5)));
    t *= 2;

    V const r[results] = {
        simd::fmadd(x, y, x),
        simd::fmsub(x, y, y),
        simd::fnmadd(x, y, x),
        simd::fnmsub(y, x, y),
        simd::div(x, y),
        simd::sqrt(x),
        simd::min(x, y),
        simd::max(x, y),
        simd::abs(x),
        simd::floor(x),
        simd::ceil(x),
        simd::trunc(x),
        simd::round(x),
        simd::round_even(x),
        simd::exp(x),
        simd::erf(x),
        simd::erfc(x),
        simd::rsqrt(x),
        simd::select(simd::cmp_lt(x, y), x, y),
        simd::convert<S>(simd::convert<I>(x)),
        simd::bitcast<S>(simd::shift_left<1>(simd::bitcast<I>(x))),
        simd::lookup(table, idx),
        simd::lookup(table, idx, simd::cmp_lt(x, y)),
        simd::select(simd::cmp_lt(t, x), simd::lookup(table, idx), simd::sqrt(t)),
        x * y + x, // a multiply and add of the kernel's own, kept apart by --fmad=false
    };
    for (int k = 0; k < results; ++k) {
        out[k * n + i] = r[k];
    }
}

template <typename S>
__global__ void run(S const *x, S const *y, S const *table, int table_size, S *out, int n) {
    int const i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        operations(x[i], y[i], table, table_size, out, n, i);
    }
}

/// Every pair from values that reach each operation's special cases: NaN, infinities, signed zeros,
/// subnormals, rounding ties, each piece of erf and erfc, and both ends of exp's range.
template <typename S>
void problems(std::vector<S> &x, std::vector<S> &y) {
    S const              nan = std::numeric_limits<S>::quiet_NaN();
    S const              inf = std::numeric_limits<S>::infinity();
    std::vector<S> const values{S(0),       S(-0.0), S(1),     S(-1),  S(0.5),  S(-0.5), S(1.5),
                                S(-1.5),    S(2.5),  S(-3.25), nan,    inf,     -inf,    std::numeric_limits<S>::denorm_min(),
                                S(1e-30),   S(0.3),  S(-0.7),  S(1.9), S(3.3),  S(-5),   S(9.5),
                                S(27),      S(-80),  S(85),    S(700), S(-750), S(1e30), std::nextafter(S(0.5), S(0)),
                                S(123456.5)};
    for (S a : values) {
        for (S b : values) {
            x.push_back(a);
            y.push_back(b);
        }
    }
}

template <typename S>
bool same(S a, S b) {
    if (std::isnan(a) && std::isnan(b)) {
        return true; // a GPU's NaN payload need not be the CPU's
    }
    return std::memcmp(&a, &b, sizeof(S)) == 0;
}

#define CHECK_CUDA(call)                                                                                                                   \
    do {                                                                                                                                   \
        cudaError_t const err_ = (call);                                                                                                   \
        if (err_ != cudaSuccess) {                                                                                                         \
            std::fprintf(stderr, "%s failed: %s\n", #call, cudaGetErrorString(err_));                                                      \
            return false;                                                                                                                  \
        }                                                                                                                                  \
    } while (false)

template <typename S>
bool check(char const *name) {
    std::vector<S> x, y;
    problems(x, y);
    int const n = static_cast<int>(x.size());

    std::vector<S> table(64);
    for (std::size_t k = 0; k < table.size(); ++k) {
        table[k] = S(1) / S(k + 3);
    }
    int const table_size = static_cast<int>(table.size());

    std::vector<S> want(static_cast<std::size_t>(results) * x.size());
    for (int i = 0; i < n; ++i) {
        operations(x[i], y[i], table.data(), table_size, want.data(), n, i);
    }

    S *dx = nullptr, *dy = nullptr, *dtable = nullptr, *dout = nullptr;
    CHECK_CUDA(cudaMalloc(&dx, x.size() * sizeof(S)));
    CHECK_CUDA(cudaMalloc(&dy, y.size() * sizeof(S)));
    CHECK_CUDA(cudaMalloc(&dtable, table.size() * sizeof(S)));
    CHECK_CUDA(cudaMalloc(&dout, want.size() * sizeof(S)));
    CHECK_CUDA(cudaMemcpy(dx, x.data(), x.size() * sizeof(S), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dy, y.data(), y.size() * sizeof(S), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dtable, table.data(), table.size() * sizeof(S), cudaMemcpyHostToDevice));
    run<<<(n + 127) / 128, 128>>>(dx, dy, dtable, table_size, dout, n);
    CHECK_CUDA(cudaGetLastError());
    std::vector<S> got(want.size());
    CHECK_CUDA(cudaMemcpy(got.data(), dout, got.size() * sizeof(S), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaFree(dx));
    CHECK_CUDA(cudaFree(dy));
    CHECK_CUDA(cudaFree(dtable));
    CHECK_CUDA(cudaFree(dout));

    int mismatches = 0;
    for (int k = 0; k < results; ++k) {
        for (int i = 0; i < n; ++i) {
            std::size_t const at = static_cast<std::size_t>(k) * x.size() + static_cast<std::size_t>(i);
            if (!same(got[at], want[at])) {
                if (++mismatches <= 20) {
                    std::fprintf(stderr, "%s result %d: x = %a, y = %a: device %a, host %a\n", name, k, double(x[i]), double(y[i]),
                                 double(got[at]), double(want[at]));
                }
            }
        }
    }
    std::printf("%s: %d results, %d mismatches\n", name, results * n, mismatches);
    return mismatches == 0;
}

} // namespace

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
        std::printf("no CUDA device: the kernels compiled, nothing to run\n");
        return 77;
    }
    bool const ok_double = check<double>("double");
    bool const ok_float  = check<float>("float");
    return ok_double && ok_float ? 0 : 1;
}
