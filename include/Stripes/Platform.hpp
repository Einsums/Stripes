//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Stripes/Config.hpp>
#include <cstddef>
#include <cstdint>

// ---------------------------------------------------------------------------
// The instruction-set namespace.
//
// Everything in these headers compiles differently for each set of target
// features: Vec<float> is a different register, loadu a different
// instruction, native_lanes<float> a different number. A program can hold
// translation units built for several such sets (the per-rung kernels of
// stripes_add_dispatch_sources, or a user's -march=native TU next to the
// library's baseline), and if those definitions shared names, the linker
// would keep one copy of each and hand it to every caller. Nothing would
// break while every call inlines, which is why it went unseen; an
// out-of-line copy (an -O0 build, a refused __forceinline, a taken address)
// is then called with the wrong register width, or with instructions the
// CPU lacks.
//
// So the headers declare everything inside an inline namespace named for
// the features the translation unit was compiled with, isa_<tier>_<ext>...
// It is transparent to lookup, so stripes::Vec<float> still names it,
// but each feature set mangles differently and no two can be merged.
//
// The name has to change whenever any definition does, so it encodes every
// feature macro the headers test. devtools/check_isa_tag.py fails
// pre-commit when a header tests a macro that is not named between the
// markers below; add a piece here for it.
//
// RuntimeFeatures.hpp and RungLadder.hpp describe the machine
// at run time, compile identically everywhere, and stay in stripes.
// ---------------------------------------------------------------------------

// BEGIN ISA TAG
// FMA3 is usable in this translation unit. GCC, Clang, clang-cl and icx
// define __FMA__ whenever FMA3 is enabled. The true MSVC driver never does,
// not even under /arch:AVX2 or /arch:AVX512, although both of those enable
// FMA3 code generation and its intrinsics, so key MSVC off __AVX2__ (which
// /arch:AVX512 also defines).
#if defined(__FMA__) || (defined(_MSC_VER) && !defined(__clang__) && defined(__AVX2__))
#    define STRIPES_HAVE_FMA 1
#endif

#if defined(__AVX512F__) && defined(__AVX512VL__)
#    define STRIPES_ISA_TIER_ isa_avx512
#elif defined(__AVX__)
#    define STRIPES_ISA_TIER_ isa_avx
#elif defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#    define STRIPES_ISA_TIER_ isa_sse2
#elif defined(__aarch64__) || defined(_M_ARM64)
#    define STRIPES_ISA_TIER_ isa_neon
#else
#    define STRIPES_ISA_TIER_       isa_scalar
/// No SIMD unit: every Vec<T> is one T. Defined for code that has to leave out what needs more than
/// one lane per register, such as CVec, at the preprocessor.
#    define STRIPES_SCALAR_FALLBACK 1
#endif

// One piece per feature: its suffix when the feature is on, nothing when it is off.
#if defined(__AVX512F__)
#    define STRIPES_ISA_P01_ _avx512f
#else
#    define STRIPES_ISA_P01_
#endif
#if defined(__AVX2__)
#    define STRIPES_ISA_P02_ _avx2
#else
#    define STRIPES_ISA_P02_
#endif
#if defined(STRIPES_HAVE_FMA)
#    define STRIPES_ISA_P03_ _fma
#else
#    define STRIPES_ISA_P03_
#endif
#if defined(__SSE3__)
#    define STRIPES_ISA_P04_ _sse3
#else
#    define STRIPES_ISA_P04_
#endif
#if defined(__SSSE3__)
#    define STRIPES_ISA_P05_ _ssse3
#else
#    define STRIPES_ISA_P05_
#endif
#if defined(__SSE4_1__)
#    define STRIPES_ISA_P06_ _sse41
#else
#    define STRIPES_ISA_P06_
#endif
#if defined(__SSE4_2__)
#    define STRIPES_ISA_P07_ _sse42
#else
#    define STRIPES_ISA_P07_
#endif
#if defined(__PRFCHW__) || defined(__3dNOW__)
#    define STRIPES_ISA_P08_ _prfchw
#else
#    define STRIPES_ISA_P08_
#endif
#if defined(__AVX512DQ__)
#    define STRIPES_ISA_P09_ _dq
#else
#    define STRIPES_ISA_P09_
#endif
#if defined(__AVX512FP16__)
#    define STRIPES_ISA_P10_ _fp16
#else
#    define STRIPES_ISA_P10_
#endif
#if defined(__AVX512BF16__)
#    define STRIPES_ISA_P11_ _bf16
#else
#    define STRIPES_ISA_P11_
#endif
#if defined(__AVX512VNNI__)
#    define STRIPES_ISA_P12_ _vnni
#else
#    define STRIPES_ISA_P12_
#endif
#if defined(__AVXVNNI__)
#    define STRIPES_ISA_P13_ _avxvnni
#else
#    define STRIPES_ISA_P13_
#endif
#if defined(__AVX10_1__)
#    define STRIPES_ISA_P14_ _avx10v1
#else
#    define STRIPES_ISA_P14_
#endif
#if defined(__AVX10_2__)
#    define STRIPES_ISA_P15_ _avx10v2
#else
#    define STRIPES_ISA_P15_
#endif
#if defined(__AVX10_1_256__) || defined(__AVX10_2_256__)
#    define STRIPES_ISA_P16_ _avx10w256
#else
#    define STRIPES_ISA_P16_
#endif
#if defined(__AVX10_1_512__) || defined(__AVX10_2_512__)
#    define STRIPES_ISA_P17_ _avx10w512
#else
#    define STRIPES_ISA_P17_
#endif
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#    define STRIPES_ISA_P18_ _neonfp16
#else
#    define STRIPES_ISA_P18_
#endif
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
#    define STRIPES_ISA_P19_ _neonbf16
#else
#    define STRIPES_ISA_P19_
#endif
#if defined(__ARM_FEATURE_DOTPROD)
#    define STRIPES_ISA_P20_ _dotprod
#else
#    define STRIPES_ISA_P20_
#endif
#if defined(__ARM_FEATURE_MATMUL_INT8)
#    define STRIPES_ISA_P21_ _i8mm
#else
#    define STRIPES_ISA_P21_
#endif
// END ISA TAG

#define STRIPES_ISA_PASTE_(t, p01, p02, p03, p04, p05, p06, p07, p08, p09, p10, p11, p12, p13, p14, p15, p16, p17, p18, p19, p20, p21)     \
    t##p01##p02##p03##p04##p05##p06##p07##p08##p09##p10##p11##p12##p13##p14##p15##p16##p17##p18##p19##p20##p21
#define STRIPES_ISA_EXPAND_(...) STRIPES_ISA_PASTE_(__VA_ARGS__)

/// The inline namespace for this translation unit's features, such as isa_avx_avx2_fma_sse3_ssse3_sse41_sse42.
#define STRIPES_ISA_NS                                                                                                                     \
    STRIPES_ISA_EXPAND_(STRIPES_ISA_TIER_, STRIPES_ISA_P01_, STRIPES_ISA_P02_, STRIPES_ISA_P03_, STRIPES_ISA_P04_, STRIPES_ISA_P05_,       \
                        STRIPES_ISA_P06_, STRIPES_ISA_P07_, STRIPES_ISA_P08_, STRIPES_ISA_P09_, STRIPES_ISA_P10_, STRIPES_ISA_P11_,        \
                        STRIPES_ISA_P12_, STRIPES_ISA_P13_, STRIPES_ISA_P14_, STRIPES_ISA_P15_, STRIPES_ISA_P16_, STRIPES_ISA_P17_,        \
                        STRIPES_ISA_P18_, STRIPES_ISA_P19_, STRIPES_ISA_P20_, STRIPES_ISA_P21_)

/// Open and close the instruction-set namespace, just inside STRIPES_NAMESPACE_BEGIN().
#define STRIPES_ISA_NAMESPACE_BEGIN() inline namespace STRIPES_ISA_NS {
#define STRIPES_ISA_NAMESPACE_END()   }

STRIPES_NAMESPACE_BEGIN()
STRIPES_ISA_NAMESPACE_BEGIN()

// ---------------------------------------------------------------------------
// Feature detection: constexpr booleans for each ISA extension.
// These are independent: e.g. has_avx2 and has_sse42 are both true on AVX2 hw.
// ---------------------------------------------------------------------------

// ---- x86 features ----

inline constexpr bool has_sse2 =
#if defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
    true;
#else
    false;
#endif

inline constexpr bool has_ssse3 =
#if defined(__SSSE3__)
    true;
#else
    false;
#endif

inline constexpr bool has_sse41 =
#if defined(__SSE4_1__)
    true;
#else
    false;
#endif

inline constexpr bool has_sse42 =
#if defined(__SSE4_2__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx =
#if defined(__AVX__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx2 =
#if defined(__AVX2__)
    true;
#else
    false;
#endif

inline constexpr bool has_fma =
#if defined(STRIPES_HAVE_FMA)
    true;
#else
    false;
#endif

inline constexpr bool has_avx512 =
#if defined(__AVX512F__) && defined(__AVX512VL__)
    true;
#else
    false;
#endif

// AVX-10 (Intel's consolidation of AVX-512 under a version number). AVX10.2
// requires 512-bit vectors on every part, and the parts that ship AVX10
// (Granite Rapids and later) also set the legacy AVX-512 CPUID bits and
// define `__AVX512F__`, so they already take the AVX-512 tier in Operations
// and Shuffle and the V4 dispatch rung. A 256-bit-only AVX10 part would
// define `__AVX2__` without `__AVX512F__` and take the AVX2 tier, which is
// correct for it. These flags only expose the AVX10 version to callers that
// want to hand-write kernels for it.
inline constexpr bool has_avx10_1 =
#if defined(__AVX10_1__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx10_2 =
#if defined(__AVX10_2__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx10_256 =
#if defined(__AVX10_1_256__) || defined(__AVX10_2_256__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx10_512 =
#if defined(__AVX10_1_512__) || defined(__AVX10_2_512__)
    true;
#else
    false;
#endif

// VNNI (Vector Neural Network Instructions): int8 dot-product accelerators.
//   - `has_avx_vnni`     ⇒ Alder Lake+ consumer chips, 256-bit dpbusd_epi32.
//   - `has_avx512_vnni`  ⇒ Cascade Lake-X+ server, 512-bit dpbusd at full
//     AVX-512 width. Both add the fused unsigned×signed → int32
//     accumulator used by quantized ML kernels.
inline constexpr bool has_avx_vnni =
#if defined(__AVXVNNI__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx512_vnni =
#if defined(__AVX512VNNI__)
    true;
#else
    false;
#endif

// ---- ARM features ----

inline constexpr bool has_neon =
#if defined(__aarch64__) || defined(_M_ARM64)
    true;
#else
    false;
#endif

inline constexpr bool has_neon_fp16 =
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    true;
#else
    false;
#endif

inline constexpr bool has_neon_bf16 =
#if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
    true;
#else
    false;
#endif

inline constexpr bool has_neon_i8mm =
#if defined(__ARM_FEATURE_MATMUL_INT8)
    true;
#else
    false;
#endif

// FEAT_DOTPROD: vdotq_s32/_u32. Default-on for M1+ and most modern
// aarch64 server chips; gates the same-sign int8 dot-product helpers.
inline constexpr bool has_neon_dotprod =
#if defined(__ARM_FEATURE_DOTPROD)
    true;
#else
    false;
#endif

// ---- x86 half-precision features (AVX-512 sub-extensions) ----
//
// `has_avx512_fp16` ⇒ Sapphire Rapids+ (consumer: not yet shipping).
// `has_avx512_bf16` ⇒ Cooper Lake+ (server) and Granite Rapids client.
// Both gate native arithmetic on the respective half-format; without
// them, kernels must round-trip through FP32.

inline constexpr bool has_avx512_fp16 =
#if defined(__AVX512FP16__)
    true;
#else
    false;
#endif

inline constexpr bool has_avx512_bf16 =
#if defined(__AVX512BF16__)
    true;
#else
    false;
#endif

inline constexpr bool is_apple_silicon =
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__APPLE__)
    true;
#else
    false;
#endif

// ---------------------------------------------------------------------------
// Native register width: the widest available ISA determines this.
// Used to set Vec<T>::lanes and blocking parameters.
// ---------------------------------------------------------------------------

#if defined(__AVX512F__) && defined(__AVX512VL__)
inline constexpr int native_bits = 512;
#elif defined(__AVX__)
inline constexpr int native_bits = 256;
#elif defined(__SSE2__) || (defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2))
                inline constexpr int native_bits = 128;
#elif defined(__aarch64__) || defined(_M_ARM64)
                inline constexpr int native_bits = 128;
#else
                inline constexpr int native_bits = 0; // scalar fallback
#endif

inline constexpr int    native_bytes     = (native_bits > 0) ? (native_bits / 8) : 0;
inline constexpr size_t native_alignment = (native_bytes > 0) ? static_cast<size_t>(native_bytes) : alignof(double);

/// Number of elements of type T that fit in one native SIMD register.
/// Returns 1 for scalar fallback so loop counts remain valid.
template <typename T>
inline constexpr int native_lanes = (native_bits > 0) ? (native_bits / (8 * static_cast<int>(sizeof(T)))) : 1;

STRIPES_ISA_NAMESPACE_END()
STRIPES_NAMESPACE_END()
