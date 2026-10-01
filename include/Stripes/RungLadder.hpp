//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

/**
 * @file RungLadder.hpp
 * @brief The per-rung entry points of a kernel compiled once per rung, as a @ref stripes::Ladder for @ref stripes::select.
 *
 * A module whose kernel translation units go through ``stripes_add_dispatch_sources``
 * compiles one copy per rung, each in its own ``arch_<rung>`` namespace, and the target is told
 * which rungs it built through ``STRIPES_HAS_RUNG_<RUNG>``. These macros turn that set into
 * the two things every dispatching module needs:
 *
 * @code
 * // Declare the entry in each namespace that was built.
 * #define DECLARE(ns) namespace ns { void kernel(float const *, float *, std::size_t); }
 * STRIPES_FOR_EACH_BUILT_RUNG(DECLARE)
 * #undef DECLARE
 *
 * // Pick the best one for this machine.
 * static KernelFn const kernel = stripes::select<KernelFn>(STRIPES_LADDER(kernel));
 * @endcode
 *
 * A rung that was not built fills its slot with nullptr, which @ref stripes::select falls
 * through. ``arch_native`` (single-TU aarch64 builds and pinned targets) occupies the baseline
 * slot: on aarch64 it IS the toolchain baseline, so an opt-in SME rung can still out-rank it
 * through the ordinary walk.
 */

#include <Stripes/RuntimeFeatures.hpp>

#if defined(STRIPES_HAS_RUNG_NATIVE)
#    define STRIPES_BASELINE_SLOT(fn) &arch_native::fn
#    define STRIPES_EACH_NATIVE(X)    X(arch_native)
#elif defined(STRIPES_HAS_RUNG_BASELINE)
#    define STRIPES_BASELINE_SLOT(fn) &arch_baseline::fn
#    define STRIPES_EACH_NATIVE(X)
#else
#    define STRIPES_BASELINE_SLOT(fn) nullptr
#    define STRIPES_EACH_NATIVE(X)
#endif

#if defined(STRIPES_HAS_RUNG_BASELINE)
#    define STRIPES_EACH_BASELINE(X) X(arch_baseline)
#else
#    define STRIPES_EACH_BASELINE(X)
#endif

#if defined(STRIPES_HAS_RUNG_V2)
#    define STRIPES_V2_SLOT(fn) &arch_v2::fn
#    define STRIPES_EACH_V2(X)  X(arch_v2)
#else
#    define STRIPES_V2_SLOT(fn) nullptr
#    define STRIPES_EACH_V2(X)
#endif

#if defined(STRIPES_HAS_RUNG_V3)
#    define STRIPES_V3_SLOT(fn) &arch_v3::fn
#    define STRIPES_EACH_V3(X)  X(arch_v3)
#else
#    define STRIPES_V3_SLOT(fn) nullptr
#    define STRIPES_EACH_V3(X)
#endif

#if defined(STRIPES_HAS_RUNG_V4)
#    define STRIPES_V4_SLOT(fn) &arch_v4::fn
#    define STRIPES_EACH_V4(X)  X(arch_v4)
#else
#    define STRIPES_V4_SLOT(fn) nullptr
#    define STRIPES_EACH_V4(X)
#endif

#if defined(STRIPES_HAS_RUNG_SME)
#    define STRIPES_SME_SLOT(fn) &arch_sme::fn
#    define STRIPES_EACH_SME(X)  X(arch_sme)
#else
#    define STRIPES_SME_SLOT(fn) nullptr
#    define STRIPES_EACH_SME(X)
#endif

#if defined(STRIPES_HAS_RUNG_SVE128)
#    define STRIPES_SVE128_SLOT(fn) &arch_sve128::fn
#    define STRIPES_EACH_SVE128(X)  X(arch_sve128)
#else
#    define STRIPES_SVE128_SLOT(fn) nullptr
#    define STRIPES_EACH_SVE128(X)
#endif

#if defined(STRIPES_HAS_RUNG_SVE256)
#    define STRIPES_SVE256_SLOT(fn) &arch_sve256::fn
#    define STRIPES_EACH_SVE256(X)  X(arch_sve256)
#else
#    define STRIPES_SVE256_SLOT(fn) nullptr
#    define STRIPES_EACH_SVE256(X)
#endif

#if defined(STRIPES_HAS_RUNG_SVE512)
#    define STRIPES_SVE512_SLOT(fn) &arch_sve512::fn
#    define STRIPES_EACH_SVE512(X)  X(arch_sve512)
#else
#    define STRIPES_SVE512_SLOT(fn) nullptr
#    define STRIPES_EACH_SVE512(X)
#endif

/// Invoke @p X with the name of every ``arch_<rung>`` namespace this target built.
// One rung per line; clang-format otherwise reflows this differently on every run.
// clang-format off
#define STRIPES_FOR_EACH_BUILT_RUNG(X) \
    STRIPES_EACH_NATIVE(X)             \
    STRIPES_EACH_BASELINE(X)           \
    STRIPES_EACH_V2(X)                 \
    STRIPES_EACH_V3(X)                 \
    STRIPES_EACH_V4(X)                 \
    STRIPES_EACH_SME(X)                \
    STRIPES_EACH_SVE128(X)             \
    STRIPES_EACH_SVE256(X)             \
    STRIPES_EACH_SVE512(X)
// clang-format on

/// What this target's sme copies need besides SME, as stripes_add_dispatch_sources() probed it with
/// this target's compiler; without the probe, the worst case (see stripes::SmeRungRequires).
#if defined(STRIPES_SME_RUNG_ENABLES_SVE) && defined(STRIPES_SME_RUNG_ENABLES_SVE2)
#    define STRIPES_SME_REQUIRES (::stripes::SmeRungRequires{STRIPES_SME_RUNG_ENABLES_SVE != 0, STRIPES_SME_RUNG_ENABLES_SVE2 != 0})
#else
#    define STRIPES_SME_REQUIRES (::stripes::SmeRungRequires{})
#endif

/// A stripes::Ladder of @p fn's built copies and the sme copy's requirements, for stripes::select:
/// ``stripes::select<Fn>(STRIPES_LADDER(fn))``. A rung that was not built is nullptr.
#define STRIPES_LADDER(fn)                                                                                                                 \
    {.baseline     = STRIPES_BASELINE_SLOT(fn),                                                                                            \
     .v2           = STRIPES_V2_SLOT(fn),                                                                                                  \
     .v3           = STRIPES_V3_SLOT(fn),                                                                                                  \
     .v4           = STRIPES_V4_SLOT(fn),                                                                                                  \
     .sme          = STRIPES_SME_SLOT(fn),                                                                                                 \
     .sve128       = STRIPES_SVE128_SLOT(fn),                                                                                              \
     .sve256       = STRIPES_SVE256_SLOT(fn),                                                                                              \
     .sve512       = STRIPES_SVE512_SLOT(fn),                                                                                              \
     .sme_requires = STRIPES_SME_REQUIRES}
