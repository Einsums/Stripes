//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

// ---------------------------------------------------------------------------
// The release namespace.
//
// Everything lives in stripes::v1 (inline, so stripes::Vec names it). Two
// libraries in one process may each be built against a different Stripes
// release, and the header templates would otherwise share mangled names with
// different bodies, the linker keeping one for both. The instruction-set
// namespace inside it (Platform.hpp) keeps the rungs of one release apart in
// the same way. Bump the version piece whenever a release changes a
// definition incompatibly.
// ---------------------------------------------------------------------------

#define STRIPES_VERSION_MAJOR 1
#define STRIPES_VERSION_MINOR 0
#define STRIPES_VERSION_PATCH 0

#define STRIPES_NAMESPACE_BEGIN()                                                                                                          \
    namespace stripes {                                                                                                                    \
    inline namespace v1 {
#define STRIPES_NAMESPACE_END()                                                                                                            \
    }                                                                                                                                      \
    }

/// Marks a function to be inlined even where the optimizer would decline.
#if !defined(STRIPES_FORCEINLINE)
#    if defined(__NVCC__) || defined(__CUDACC__)
#        define STRIPES_FORCEINLINE inline
#    elif defined(_MSC_VER) // MSVC and clang-cl; clang-cl defines no __GNUC__
#        define STRIPES_FORCEINLINE __forceinline
#    elif defined(__GNUC__)
#        define STRIPES_FORCEINLINE inline __attribute__((__always_inline__))
#    else
#        define STRIPES_FORCEINLINE inline
#    endif
#endif

/// Marks a function of the runtime library (RuntimeFeatures.cpp) exported from it, or imported by its
/// users. The build defines STRIPES_EXPORTS while compiling the library and STRIPES_STATIC when the
/// library is static, in which case there is nothing to export or import.
#if defined(STRIPES_STATIC)
#    define STRIPES_EXPORT
#elif defined(_WIN32) || defined(__CYGWIN__)
#    if defined(STRIPES_EXPORTS)
#        define STRIPES_EXPORT __declspec(dllexport)
#    else
#        define STRIPES_EXPORT __declspec(dllimport)
#    endif
#elif defined(__GNUC__)
#    define STRIPES_EXPORT __attribute__((visibility("default")))
#else
#    define STRIPES_EXPORT
#endif
