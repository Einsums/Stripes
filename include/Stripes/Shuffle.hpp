//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Stripes/ComplexVec.hpp>
#include <Stripes/Config.hpp>
#include <Stripes/Operations.hpp>
#include <Stripes/Partial.hpp>
#include <Stripes/Vec.hpp>
#include <algorithm>
#include <complex>
#include <cstddef>
#include <utility>

STRIPES_NAMESPACE_BEGIN()
STRIPES_ISA_NAMESPACE_BEGIN()

// ===========================================================================
// In-register transpose: transpose_inplace(Vec<T> *rows)
//
// Transposes an N×N block where N = Vec<T>::lanes. The caller passes a
// pointer to N Vec<T> values representing the rows of the block. On return,
// rows[i] contains what was column i of the original block.
//
// The block size adapts to the native register width automatically:
//   AVX-512 float: 16×16    AVX-512 double: 8×8
//   AVX float:     8×8      AVX double:     4×4
//   SSE2 float:    4×4      SSE2 double:    2×2
//   NEON float:    4×4      NEON double:    2×2
//   Scalar:        1×1 (no-op)
// ===========================================================================

// ---------------------------------------------------------------------------
// x86 AVX-512: 512-bit registers
// ---------------------------------------------------------------------------
#if defined(__AVX512F__) && defined(__AVX512VL__)

// 8×8 double transpose (AVX-512)
//
// Three-phase algorithm operating on 8 __m512d registers, structurally the
// same as the 16×16 float kernel below:
//   Phase 1: unpacklo/hi transposes the 2×2 block inside each 128-bit lane,
//            so t[2g] and t[2g+1] carry the even and odd columns of rows
//            2g and 2g+1, one column pair per lane.
//   Phase 2: shuffle_f64x2 pairs those registers at distance TWO, gathering
//            the lanes of four consecutive rows.
//   Phase 3: shuffle_f64x2 merges the two 4-row groups into whole columns.
//
// The pairing distance is what makes this correct. Pairing at distance four
// in phase 2 also produces eight plausible-looking registers, but they carry
// src[p[j]][p[i]] with p = {0,1,4,5,2,3,6,7}: the middle two 128-bit lanes
// end up swapped in both the element order and the row assignment. That
// permutation is its own inverse and fixes the identity matrix, so a
// round-trip or identity test cannot see it; only an element-by-element
// comparison against the transpose can.
STRIPES_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    // Phase 1: 2×2 transposes within 128-bit lanes.
    Vec<double> t[8]; // NOLINT
    for (int i = 0; i < 8; i += 2) {
        t[i]     = _mm512_unpacklo_pd(rows[i], rows[i + 1]);
        t[i + 1] = _mm512_unpackhi_pd(rows[i], rows[i + 1]);
    }

    // Phases 2+3: k = 0 assembles the even columns, k = 1 the odd ones.
    for (int k = 0; k < 2; k++) {
        // Phase 2: merge row-groups A(0-1)+B(2-3) and C(4-5)+D(6-7).
        auto uAB_lo = _mm512_shuffle_f64x2(t[k], t[k + 2], 0x88);
        auto uAB_hi = _mm512_shuffle_f64x2(t[k], t[k + 2], 0xdd);
        auto uCD_lo = _mm512_shuffle_f64x2(t[k + 4], t[k + 6], 0x88);
        auto uCD_hi = _mm512_shuffle_f64x2(t[k + 4], t[k + 6], 0xdd);

        // Phase 3: merge AB+CD into complete columns.
        rows[k]     = _mm512_shuffle_f64x2(uAB_lo, uCD_lo, 0x88); // column k
        rows[k + 2] = _mm512_shuffle_f64x2(uAB_hi, uCD_hi, 0x88); // column k+2
        rows[k + 4] = _mm512_shuffle_f64x2(uAB_lo, uCD_lo, 0xdd); // column k+4
        rows[k + 6] = _mm512_shuffle_f64x2(uAB_hi, uCD_hi, 0xdd); // column k+6
    }
}

// 16×16 float transpose (AVX-512)
//
// Four-phase algorithm operating on 16 __m512 registers:
//   Phase 1: unpacklo/hi interleaves pairs within 128-bit lanes
//   Phase 2: shuffle_ps groups 4 elements within each 128-bit lane
//   Phase 3: shuffle_f32x4 permutes 128-bit lanes, combining 4-row blocks into 8-row blocks
//   Phase 4: shuffle_f32x4 permutes 128-bit lanes, combining 8-row blocks into the 16-row result
//
// After phases 1+2, each register s[k] contains, in its four 128-bit lanes,
// 4 elements of column (k, k+4, k+8, k+12) from a group of 4 consecutive rows.
// Phases 3+4 reassemble these lane fragments into complete 16-element columns.
STRIPES_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    // Solve the 2x2 diagonal blocks.
    // unpack doesn't work like it does for doubles.
    // Set up for the 2x2 solve.
    __m512 const row0_iter1  = _mm512_unpacklo_ps(rows[0], rows[1]);
    __m512 const row1_iter1  = _mm512_unpackhi_ps(rows[0], rows[1]);
    __m512 const row2_iter1  = _mm512_unpacklo_ps(rows[2], rows[3]);
    __m512 const row3_iter1  = _mm512_unpackhi_ps(rows[2], rows[3]);
    __m512 const row4_iter1  = _mm512_unpacklo_ps(rows[4], rows[5]);
    __m512 const row5_iter1  = _mm512_unpackhi_ps(rows[4], rows[5]);
    __m512 const row6_iter1  = _mm512_unpacklo_ps(rows[6], rows[7]);
    __m512 const row7_iter1  = _mm512_unpackhi_ps(rows[6], rows[7]);
    __m512 const row8_iter1  = _mm512_unpacklo_ps(rows[8], rows[9]);
    __m512 const row9_iter1  = _mm512_unpackhi_ps(rows[8], rows[9]);
    __m512 const row10_iter1 = _mm512_unpacklo_ps(rows[10], rows[11]);
    __m512 const row11_iter1 = _mm512_unpackhi_ps(rows[10], rows[11]);
    __m512 const row12_iter1 = _mm512_unpacklo_ps(rows[12], rows[13]);
    __m512 const row13_iter1 = _mm512_unpackhi_ps(rows[12], rows[13]);
    __m512 const row14_iter1 = _mm512_unpacklo_ps(rows[14], rows[15]);
    __m512 const row15_iter1 = _mm512_unpackhi_ps(rows[14], rows[15]);

    // Actually solve the 2x2 blocks.
    __m512 const row0_iter2  = _mm512_shuffle_ps(row0_iter1, row1_iter1, 0x44);
    __m512 const row1_iter2  = _mm512_shuffle_ps(row0_iter1, row1_iter1, 0xee);
    __m512 const row2_iter2  = _mm512_shuffle_ps(row2_iter1, row3_iter1, 0x44);
    __m512 const row3_iter2  = _mm512_shuffle_ps(row2_iter1, row3_iter1, 0xee);
    __m512 const row4_iter2  = _mm512_shuffle_ps(row4_iter1, row5_iter1, 0x44);
    __m512 const row5_iter2  = _mm512_shuffle_ps(row4_iter1, row5_iter1, 0xee);
    __m512 const row6_iter2  = _mm512_shuffle_ps(row6_iter1, row7_iter1, 0x44);
    __m512 const row7_iter2  = _mm512_shuffle_ps(row6_iter1, row7_iter1, 0xee);
    __m512 const row8_iter2  = _mm512_shuffle_ps(row8_iter1, row9_iter1, 0x44);
    __m512 const row9_iter2  = _mm512_shuffle_ps(row8_iter1, row9_iter1, 0xee);
    __m512 const row10_iter2 = _mm512_shuffle_ps(row10_iter1, row11_iter1, 0x44);
    __m512 const row11_iter2 = _mm512_shuffle_ps(row10_iter1, row11_iter1, 0xee);
    __m512 const row12_iter2 = _mm512_shuffle_ps(row12_iter1, row13_iter1, 0x44);
    __m512 const row13_iter2 = _mm512_shuffle_ps(row12_iter1, row13_iter1, 0xee);
    __m512 const row14_iter2 = _mm512_shuffle_ps(row14_iter1, row15_iter1, 0x44);
    __m512 const row15_iter2 = _mm512_shuffle_ps(row14_iter1, row15_iter1, 0xee);

    // Solve the 4x4 diagonal blocks.
    __m512 const row0_iter3  = _mm512_shuffle_ps(row0_iter2, row2_iter2, 0x44);
    __m512 const row1_iter3  = _mm512_shuffle_ps(row1_iter2, row3_iter2, 0x44);
    __m512 const row2_iter3  = _mm512_shuffle_ps(row0_iter2, row2_iter2, 0xee);
    __m512 const row3_iter3  = _mm512_shuffle_ps(row1_iter2, row3_iter2, 0xee);
    __m512 const row4_iter3  = _mm512_shuffle_ps(row4_iter2, row6_iter2, 0x44);
    __m512 const row5_iter3  = _mm512_shuffle_ps(row5_iter2, row7_iter2, 0x44);
    __m512 const row6_iter3  = _mm512_shuffle_ps(row4_iter2, row6_iter2, 0xee);
    __m512 const row7_iter3  = _mm512_shuffle_ps(row5_iter2, row7_iter2, 0xee);
    __m512 const row8_iter3  = _mm512_shuffle_ps(row8_iter2, row10_iter2, 0x44);
    __m512 const row9_iter3  = _mm512_shuffle_ps(row9_iter2, row11_iter2, 0x44);
    __m512 const row10_iter3 = _mm512_shuffle_ps(row8_iter2, row10_iter2, 0xee);
    __m512 const row11_iter3 = _mm512_shuffle_ps(row9_iter2, row11_iter2, 0xee);
    __m512 const row12_iter3 = _mm512_shuffle_ps(row12_iter2, row14_iter2, 0x44);
    __m512 const row13_iter3 = _mm512_shuffle_ps(row13_iter2, row15_iter2, 0x44);
    __m512 const row14_iter3 = _mm512_shuffle_ps(row12_iter2, row14_iter2, 0xee);
    __m512 const row15_iter3 = _mm512_shuffle_ps(row13_iter2, row15_iter2, 0xee);

    // Solve the 8x8 diagonal blocks.
    __m512 const row0_iter4  = _mm512_shuffle_f32x4(row0_iter3, row4_iter3, 0x88);
    __m512 const row1_iter4  = _mm512_shuffle_f32x4(row1_iter3, row5_iter3, 0x88);
    __m512 const row2_iter4  = _mm512_shuffle_f32x4(row2_iter3, row6_iter3, 0x88);
    __m512 const row3_iter4  = _mm512_shuffle_f32x4(row3_iter3, row7_iter3, 0x88);
    __m512 const row4_iter4  = _mm512_shuffle_f32x4(row0_iter3, row4_iter3, 0xdd);
    __m512 const row5_iter4  = _mm512_shuffle_f32x4(row1_iter3, row5_iter3, 0xdd);
    __m512 const row6_iter4  = _mm512_shuffle_f32x4(row2_iter3, row6_iter3, 0xdd);
    __m512 const row7_iter4  = _mm512_shuffle_f32x4(row3_iter3, row7_iter3, 0xdd);
    __m512 const row8_iter4  = _mm512_shuffle_f32x4(row8_iter3, row12_iter3, 0x88);
    __m512 const row9_iter4  = _mm512_shuffle_f32x4(row9_iter3, row13_iter3, 0x88);
    __m512 const row10_iter4 = _mm512_shuffle_f32x4(row10_iter3, row14_iter3, 0x88);
    __m512 const row11_iter4 = _mm512_shuffle_f32x4(row11_iter3, row15_iter3, 0x88);
    __m512 const row12_iter4 = _mm512_shuffle_f32x4(row8_iter3, row12_iter3, 0xdd);
    __m512 const row13_iter4 = _mm512_shuffle_f32x4(row9_iter3, row13_iter3, 0xdd);
    __m512 const row14_iter4 = _mm512_shuffle_f32x4(row10_iter3, row14_iter3, 0xdd);
    __m512 const row15_iter4 = _mm512_shuffle_f32x4(row11_iter3, row15_iter3, 0xdd);

    // Solve the rest.
    rows[0]  = _mm512_shuffle_f32x4(row0_iter4, row8_iter4, 0x88);
    rows[1]  = _mm512_shuffle_f32x4(row1_iter4, row9_iter4, 0x88);
    rows[2]  = _mm512_shuffle_f32x4(row2_iter4, row10_iter4, 0x88);
    rows[3]  = _mm512_shuffle_f32x4(row3_iter4, row11_iter4, 0x88);
    rows[4]  = _mm512_shuffle_f32x4(row4_iter4, row12_iter4, 0x88);
    rows[5]  = _mm512_shuffle_f32x4(row5_iter4, row13_iter4, 0x88);
    rows[6]  = _mm512_shuffle_f32x4(row6_iter4, row14_iter4, 0x88);
    rows[7]  = _mm512_shuffle_f32x4(row7_iter4, row15_iter4, 0x88);
    rows[8]  = _mm512_shuffle_f32x4(row0_iter4, row8_iter4, 0xdd);
    rows[9]  = _mm512_shuffle_f32x4(row1_iter4, row9_iter4, 0xdd);
    rows[10] = _mm512_shuffle_f32x4(row2_iter4, row10_iter4, 0xdd);
    rows[11] = _mm512_shuffle_f32x4(row3_iter4, row11_iter4, 0xdd);
    rows[12] = _mm512_shuffle_f32x4(row4_iter4, row12_iter4, 0xdd);
    rows[13] = _mm512_shuffle_f32x4(row5_iter4, row13_iter4, 0xdd);
    rows[14] = _mm512_shuffle_f32x4(row6_iter4, row14_iter4, 0xdd);
    rows[15] = _mm512_shuffle_f32x4(row7_iter4, row15_iter4, 0xdd);
}

// ---------------------------------------------------------------------------
// x86 AVX/AVX2: 256-bit registers
// ---------------------------------------------------------------------------
#elif defined(__AVX__)

// 4×4 double transpose (AVX)
STRIPES_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    auto const t0 = _mm256_unpacklo_pd(rows[0], rows[1]); // a0 b0 a2 b2
    auto const t1 = _mm256_unpackhi_pd(rows[0], rows[1]); // a1 b1 a3 b3
    auto const t2 = _mm256_unpacklo_pd(rows[2], rows[3]); // c0 d0 c2 d2
    auto const t3 = _mm256_unpackhi_pd(rows[2], rows[3]); // c1 d1 c3 d3
    rows[0]       = _mm256_permute2f128_pd(t0, t2, 0x20);
    rows[1]       = _mm256_permute2f128_pd(t1, t3, 0x20);
    rows[2]       = _mm256_permute2f128_pd(t0, t2, 0x31);
    rows[3]       = _mm256_permute2f128_pd(t1, t3, 0x31);
}

// 8×8 float transpose (AVX)
STRIPES_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    // Phase 1: interleave pairs
    auto const t0 = _mm256_unpacklo_ps(rows[0], rows[1]);
    auto const t1 = _mm256_unpackhi_ps(rows[0], rows[1]);
    auto const t2 = _mm256_unpacklo_ps(rows[2], rows[3]);
    auto const t3 = _mm256_unpackhi_ps(rows[2], rows[3]);
    auto const t4 = _mm256_unpacklo_ps(rows[4], rows[5]);
    auto const t5 = _mm256_unpackhi_ps(rows[4], rows[5]);
    auto const t6 = _mm256_unpacklo_ps(rows[6], rows[7]);
    auto const t7 = _mm256_unpackhi_ps(rows[6], rows[7]);

    // This is needed due to the odd interleaving properties of _mm256_unpacklo_ps.
    __m256 const row0_iter2 = _mm256_shuffle_ps(t0, t1, 0x44);
    __m256 const row1_iter2 = _mm256_shuffle_ps(t0, t1, 0xee);
    __m256 const row2_iter2 = _mm256_shuffle_ps(t2, t3, 0x44);
    __m256 const row3_iter2 = _mm256_shuffle_ps(t2, t3, 0xee);
    __m256 const row4_iter2 = _mm256_shuffle_ps(t4, t5, 0x44);
    __m256 const row5_iter2 = _mm256_shuffle_ps(t4, t5, 0xee);
    __m256 const row6_iter2 = _mm256_shuffle_ps(t6, t7, 0x44);
    __m256 const row7_iter2 = _mm256_shuffle_ps(t6, t7, 0xee);

    // Phase 2: 2×2 block shuffle
    auto const s0 = _mm256_shuffle_ps(row0_iter2, row2_iter2, 0x44);
    auto const s1 = _mm256_shuffle_ps(row1_iter2, row3_iter2, 0x44);
    auto const s2 = _mm256_shuffle_ps(row0_iter2, row2_iter2, 0xee);
    auto const s3 = _mm256_shuffle_ps(row1_iter2, row3_iter2, 0xee);
    auto const s4 = _mm256_shuffle_ps(row4_iter2, row6_iter2, 0x44);
    auto const s5 = _mm256_shuffle_ps(row5_iter2, row7_iter2, 0x44);
    auto const s6 = _mm256_shuffle_ps(row4_iter2, row6_iter2, 0xee);
    auto const s7 = _mm256_shuffle_ps(row5_iter2, row7_iter2, 0xee);

    // Phase 3: cross-lane permute
    rows[0] = _mm256_permute2f128_ps(s0, s4, 0x20);
    rows[1] = _mm256_permute2f128_ps(s1, s5, 0x20);
    rows[2] = _mm256_permute2f128_ps(s2, s6, 0x20);
    rows[3] = _mm256_permute2f128_ps(s3, s7, 0x20);
    rows[4] = _mm256_permute2f128_ps(s0, s4, 0x31);
    rows[5] = _mm256_permute2f128_ps(s1, s5, 0x31);
    rows[6] = _mm256_permute2f128_ps(s2, s6, 0x31);
    rows[7] = _mm256_permute2f128_ps(s3, s7, 0x31);
}

// ---------------------------------------------------------------------------
// x86 SSE2: 128-bit registers
// ---------------------------------------------------------------------------
#elif defined(__SSE__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)

// 2×2 double transpose (SSE2)
#    ifdef __SSE2__
STRIPES_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    auto const lo = _mm_unpacklo_pd(rows[0], rows[1]);
    auto const hi = _mm_unpackhi_pd(rows[0], rows[1]);
    rows[0]       = lo;
    rows[1]       = hi;
}
#    endif

// 4×4 float transpose (SSE)
STRIPES_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    _MM_TRANSPOSE4_PS(rows[0], rows[1], rows[2], rows[3]);
}

// ---------------------------------------------------------------------------
// ARM NEON (aarch64, including Apple Silicon): 128-bit registers
// ---------------------------------------------------------------------------
#elif defined(STRIPES_SVE_BITS)

// An L x L transpose for any L, in log2(L) rounds: each round interleaves rows i and i + L/2 (ZIP1 and
// ZIP2) into rows 2i and 2i + 1, and after the last, row j holds column j.
#    define STRIPES_SVE_TRANSPOSE(T, sfx)                                                                                                  \
        STRIPES_FORCEINLINE void transpose_inplace(Vec<T> *rows) {                                                                         \
            constexpr int L = Vec<T>::lanes;                                                                                               \
            using R         = typename Vec<T>::reg_type;                                                                                   \
            R t[L];                                                                                                                        \
            for (int i = 0; i < L; ++i)                                                                                                    \
                t[i] = rows[i].reg;                                                                                                        \
            for (int round = 1; round < L; round *= 2) {                                                                                   \
                R u[L];                                                                                                                    \
                for (int i = 0; i < L / 2; ++i) {                                                                                          \
                    u[2 * i]     = svzip1_##sfx(t[i], t[i + L / 2]);                                                                       \
                    u[2 * i + 1] = svzip2_##sfx(t[i], t[i + L / 2]);                                                                       \
                }                                                                                                                          \
                for (int i = 0; i < L; ++i)                                                                                                \
                    t[i] = u[i];                                                                                                           \
            }                                                                                                                              \
            for (int i = 0; i < L; ++i)                                                                                                    \
                rows[i].reg = t[i];                                                                                                        \
        }
STRIPES_SVE_TRANSPOSE(double, f64)
STRIPES_SVE_TRANSPOSE(float, f32)
#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
STRIPES_SVE_TRANSPOSE(half_t, f16)
#    endif
#    if defined(__ARM_FEATURE_SVE_BF16) && defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
STRIPES_SVE_TRANSPOSE(bfloat16_t, bf16)
#    endif
#    undef STRIPES_SVE_TRANSPOSE

#elif defined(__aarch64__) || defined(_M_ARM64)

// 2×2 double transpose (NEON)
STRIPES_FORCEINLINE void transpose_inplace(Vec<double> *rows) {
    auto lo = vzip1q_f64(rows[0], rows[1]);
    auto hi = vzip2q_f64(rows[0], rows[1]);
    rows[0] = lo;
    rows[1] = hi;
}

// 4×4 float transpose (NEON)
STRIPES_FORCEINLINE void transpose_inplace(Vec<float> *rows) {
    float32x4x2_t const t0 = vuzpq_f32(rows[0], rows[2]);
    float32x4x2_t const t1 = vuzpq_f32(rows[1], rows[3]);
    float32x4x2_t const t2 = vtrnq_f32(t0.val[0], t1.val[0]);
    float32x4x2_t const t3 = vtrnq_f32(t0.val[1], t1.val[1]);
    rows[0]                = t2.val[0];
    rows[1]                = t3.val[0];
    rows[2]                = t2.val[1];
    rows[3]                = t3.val[1];
}

// 8×8 NEON 16-bit transpose helper.
//
// Three TRN stages (each across 8 vector ops): TRN1/TRN2 at u16 width
// transposes 2×2 sub-blocks, then at u32 width 4×4, then at u64 width 8×8.
// The output rows arrive permuted (0,4,2,6,1,5,3,7) which we resolve at
// the call site. Total: 24 vector ops, no memory round-trip.
namespace detail {
STRIPES_FORCEINLINE void transpose_8x8_u16(uint16x8_t *r) {
    uint16x8_t const t0 = vtrn1q_u16(r[0], r[1]);
    uint16x8_t const t1 = vtrn2q_u16(r[0], r[1]);
    uint16x8_t const t2 = vtrn1q_u16(r[2], r[3]);
    uint16x8_t const t3 = vtrn2q_u16(r[2], r[3]);
    uint16x8_t const t4 = vtrn1q_u16(r[4], r[5]);
    uint16x8_t const t5 = vtrn2q_u16(r[4], r[5]);
    uint16x8_t const t6 = vtrn1q_u16(r[6], r[7]);
    uint16x8_t const t7 = vtrn2q_u16(r[6], r[7]);

    uint32x4_t const u0 = vtrn1q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
    uint32x4_t const u1 = vtrn2q_u32(vreinterpretq_u32_u16(t0), vreinterpretq_u32_u16(t2));
    uint32x4_t const u2 = vtrn1q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
    uint32x4_t const u3 = vtrn2q_u32(vreinterpretq_u32_u16(t1), vreinterpretq_u32_u16(t3));
    uint32x4_t const u4 = vtrn1q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
    uint32x4_t const u5 = vtrn2q_u32(vreinterpretq_u32_u16(t4), vreinterpretq_u32_u16(t6));
    uint32x4_t const u6 = vtrn1q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));
    uint32x4_t const u7 = vtrn2q_u32(vreinterpretq_u32_u16(t5), vreinterpretq_u32_u16(t7));

    // Stage 3: u64 TRN gives the final rows, but in shuffled order.
    // After this step, results land at (0,4,2,6,1,5,3,7), so write back at the right slots.
    r[0] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4)));
    r[4] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u0), vreinterpretq_u64_u32(u4)));
    r[2] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5)));
    r[6] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u1), vreinterpretq_u64_u32(u5)));
    r[1] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6)));
    r[5] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u2), vreinterpretq_u64_u32(u6)));
    r[3] = vreinterpretq_u16_u64(vtrn1q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7)));
    r[7] = vreinterpretq_u16_u64(vtrn2q_u64(vreinterpretq_u64_u32(u3), vreinterpretq_u64_u32(u7)));
}
} // namespace detail

#    if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
STRIPES_FORCEINLINE void transpose_inplace(Vec<half_t> *rows) {
    constexpr int N = Vec<half_t>::lanes; // 8
    uint16x8_t    u[N];                   // NOLINT
    for (int i = 0; i < N; ++i)
        u[i] = vreinterpretq_u16_f16(rows[i].reg);
    detail::transpose_8x8_u16(u);
    for (int i = 0; i < N; ++i)
        rows[i].reg = vreinterpretq_f16_u16(u[i]);
}
#    endif

#    if defined(__ARM_FEATURE_BF16_VECTOR_ARITHMETIC)
STRIPES_FORCEINLINE void transpose_inplace(Vec<bfloat16_t> *rows) {
    constexpr int N = Vec<bfloat16_t>::lanes; // 8
    uint16x8_t    u[N];                       // NOLINT
    for (int i = 0; i < N; ++i)
        u[i] = vreinterpretq_u16_bf16(rows[i].reg);
    detail::transpose_8x8_u16(u);
    for (int i = 0; i < N; ++i)
        rows[i].reg = vreinterpretq_bf16_u16(u[i]);
}
#    endif

// ---------------------------------------------------------------------------
// Scalar fallback: 1×1 is a no-op
// ---------------------------------------------------------------------------
#else

STRIPES_FORCEINLINE void transpose_inplace(Vec<float> * /*rows*/) {
    // 1×1: nothing to do
}
STRIPES_FORCEINLINE void transpose_inplace(Vec<double> * /*rows*/) {
    // 1×1: nothing to do
}

#endif

#if !defined(STRIPES_SCALAR_FALLBACK) // a one-lane register holds no complex value: no CVec
// ===========================================================================
// Complex transpose: transpose_inplace for CVec<T>
//
// Each complex<T> occupies 2*sizeof(T) bytes. So:
//   complex<float> on AVX: 4 complex values in 256 bits → 4×4 transpose
//     (same register pattern as double 4×4)
//   complex<double> on AVX: 2 complex values in 256 bits → 2×2 transpose
//     (same register pattern as double 2×2 on SSE2)
//   complex<float> on SSE2/NEON: 2 complex values → 2×2 transpose
//   complex<double> on SSE2/NEON: 1 complex value → no-op
//
// We reinterpret CVec<T> registers and delegate to the double/float
// transpose at the appropriate width.
// ===========================================================================

/// Transpose N×N block of complex<float> values in-place.
/// N = CVec<float>::complex_lanes.
STRIPES_FORCEINLINE void complex_transpose_inplace(CVec<float> *rows) {
    constexpr int N = CVec<float>::complex_lanes;
    if constexpr (N <= 1) {
        // 1×1 or 0: nothing to do
        return;
    }

    // Each complex<float> is 64 bits (same as double).
    // Reinterpret the Vec<float> registers as if they hold N "double-width"
    // elements and use the store-transpose-load approach.
    alignas(native_alignment) float buf[N * N * 2]; // N rows × N complex values × 2 floats
    for (int i = 0; i < N; ++i)
        storeu(&buf[i * N * 2], rows[i].reg);

    // Scalar transpose of complex pairs
    // buf is N×N complex matrix in row-major, each complex = 2 floats
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            // Swap complex element (i,j) with (j,i)
            std::swap(buf[(i * N + j) * 2], buf[(j * N + i) * 2]);
            std::swap(buf[(i * N + j) * 2 + 1], buf[(j * N + i) * 2 + 1]);
        }
    }

    for (int i = 0; i < N; ++i)
        rows[i].reg = loadu(&buf[i * N * 2]);
}

/// Transpose N×N block of complex<double> values in-place.
/// N = CVec<double>::complex_lanes.
STRIPES_FORCEINLINE void complex_transpose_inplace(CVec<double> *rows) {
    constexpr int N = CVec<double>::complex_lanes;
    if constexpr (N <= 1) {
        return;
    }

    alignas(native_alignment) double buf[N * N * 2];
    for (int i = 0; i < N; ++i)
        storeu(&buf[i * N * 2], rows[i].reg);

    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            std::swap(buf[(i * N + j) * 2], buf[(j * N + i) * 2]);
            std::swap(buf[(i * N + j) * 2 + 1], buf[(j * N + i) * 2 + 1]);
        }
    }

    for (int i = 0; i < N; ++i)
        rows[i].reg = loadu(&buf[i * N * 2]);
}
#endif // STRIPES_SCALAR_FALLBACK

// ===========================================================================
// Interleaved store: storeu_interleaved<R>(dst, rows)
//
// Writes R rows lane by lane, row index fastest:
//
//     dst[k * R + r] = rows[r][k]      for k < lanes, r < R
//
// exactly R * lanes elements and nothing past them. It is NEON's vst2/vst3/vst4
// for any R up to the lane count, and the store a transpose into a panel of R
// rows needs: a GEMM B panel of NR = 6 rows is 6 * lanes contiguous elements
// per register tile, fewer rows than a float register holds on AVX2 or AVX-512.
//
// The rows are padded to a full tile and transposed in registers, and column k
// is stored at dst + k * R. A full-width store there also writes the first
// lanes of the next columns, but the store of each later column rewrites
// those, so every column that ends inside the output goes out whole and only
// the last ceil(lanes / R) columns, which would end past it, use a partial
// store. A partial store is the costly one (VMASKMOV on AVX, a stack buffer on
// backends without masked stores), and this keeps it to one or two per tile.
// ===========================================================================

template <int R, typename T>
STRIPES_FORCEINLINE void storeu_interleaved(T *dst, Vec<T> const *rows) {
    constexpr int L = Vec<T>::lanes;
    static_assert(R >= 1 && R <= L, "storeu_interleaved: between one row and a full register's lanes");

    Vec<T> tile[L];
    for (int r = 0; r < R; ++r) {
        tile[r] = rows[r];
    }
    // The padding rows land only in lanes a later store rewrites; zeroing them
    // keeps the transpose from reading uninitialized registers.
    for (int r = R; r < L; ++r) {
        tile[r] = broadcast(T{0});
    }
    transpose_inplace(tile);

    // Column k's full store ends at k * R + L, inside the output while
    // k * R + L <= R * L.
    constexpr int full = L - (L + R - 1) / R + 1;
    for (int k = 0; k < full; ++k) {
        storeu(dst + k * R, tile[k]);
    }
    for (int k = full; k < L; ++k) {
        storeu_partial(dst + k * R, tile[k], static_cast<std::size_t>(R));
    }
}

// ===========================================================================
// Deinterleaving load: loadu_deinterleaved<R>(src, rows)
//
// The inverse of storeu_interleaved: reads R * lanes elements, row index
// fastest, into R rows,
//
//     rows[r][k] = src[k * R + r]      for k < lanes, r < R
//
// and nothing past them. It turns an array of structures of R fields into R
// vectors of one field each, the AoS-to-SoA step at the edge of a kernel that
// works on structure-of-arrays data. It is NEON's vld2/vld3/vld4 for any R up
// to the lane count.
//
// Column k, the R fields of element k, is loaded from src + k * R into a tile
// and the tile is transposed in registers. A full-width load there also reads
// the first fields of the next elements, which land in lanes the transpose
// sends to rows past R and are dropped, so every column that ends inside the
// input is read whole and only the last ceil(lanes / R) columns, which would
// end past it, use a partial load, as storeu_interleaved does for its stores.
// ===========================================================================

template <int R, typename T>
STRIPES_FORCEINLINE void loadu_deinterleaved(T const *src, Vec<T> *rows) {
    constexpr int L = Vec<T>::lanes;
    static_assert(R >= 1 && R <= L, "loadu_deinterleaved: between one row and a full register's lanes");

    // Column k's full load ends at k * R + L, inside the input while k * R + L <= R * L.
    constexpr int full = L - (L + R - 1) / R + 1;
    Vec<T>        tile[L];
    for (int k = 0; k < full; ++k) {
        tile[k] = loadu(src + k * R);
    }
    for (int k = full; k < L; ++k) {
        tile[k] = loadu_partial(src + k * R, static_cast<std::size_t>(R));
    }
    transpose_inplace(tile);
    for (int r = 0; r < R; ++r) {
        rows[r] = tile[r];
    }
}

STRIPES_ISA_NAMESPACE_END()
STRIPES_NAMESPACE_END()
