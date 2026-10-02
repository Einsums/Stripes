# Stripes

Portable SIMD vectors for C++20, with runtime instruction-set dispatch.

```cpp
#include <Stripes/Math.hpp>
#include <Stripes/Partial.hpp>

void scaled_exp(double const *x, double *y, std::size_t n, double a) {
    constexpr std::size_t L = stripes::lanes<double>;
    std::size_t i = 0;
    for (; i + L <= n; i += L) {
        stripes::storeu(y + i, stripes::exp(stripes::loadu(x + i) * a));
    }
    if (i < n) {
        stripes::storeu_partial(y + i, stripes::exp(stripes::loadu_partial(x + i, n - i) * a), n - i);
    }
}
```

- `Vec<T>` at the target's native width (SSE2 through AVX-512, NEON, fixed-length SVE, or a scalar fallback), wider `Vec<T, N>` for mixed precision at equal lane counts, `Mask<T>`, gathers, shuffles, reductions, and `exp`, `erf`, `erfc` and `rsqrt` that give the same bits for vectors and scalars.
- One kernel source compiled per instruction-set rung (x86-64 baseline, v2, v3, v4; aarch64 SME and SVE at 128, 256 or 512 bits) with `stripes_add_dispatch_sources()`, chosen at run time by `stripes::selected_arch()`. `STRIPES_ARCH=v2` caps the rung.
- A kernel body written over its value type also runs in CUDA kernels as its `float` or `double` instantiation, one problem per thread, and matches the CPU bit for bit under the build flags `docs/index.rst` names.
- Everything is header-only except a small runtime library for CPU detection.

## Building

```bash
cmake -S . -B build -GNinja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
ctest --test-dir build
```

Catch2 3 is found, or fetched, for the tests. Options: `STRIPES_WITH_DISPATCH` (default ON), `STRIPES_NATIVE_ARCH`, `STRIPES_TARGET_CPU`, `STRIPES_SHARED` (default ON), and `STRIPES_WITH_TESTS`, `STRIPES_WITH_EXAMPLES`, `STRIPES_WITH_BENCHMARKS` (ON when Stripes is the top-level project).

## Using it from CMake

```cmake
find_package(Stripes 1 REQUIRED)
target_link_libraries(mylib PRIVATE Stripes::stripes)

# A kernel compiled once per rung, each copy in its own arch_<rung> namespace.
stripes_add_dispatch_sources(MyKernelRungs IMPL src/MyKernelImpl.cpp)
target_sources(mylib PRIVATE ${MyKernelRungs})
target_compile_definitions(mylib PRIVATE ${MyKernelRungs_DEFINITIONS})
```

See `docs/index.rst` for the full guide and `examples/` for small checked programs.

Stripes began as the SIMD module of [Einsums](https://github.com/Einsums/Einsums).
