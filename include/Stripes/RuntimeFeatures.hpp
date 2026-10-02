//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#pragma once

#include <Stripes/Config.hpp>
#include <cstdint>
#include <optional>
#include <span>
#include <string_view>

STRIPES_NAMESPACE_BEGIN()

/**
 * @brief The instruction-set family a CpuFeatures describes.
 *
 * Each family has its own dispatch ladder (see preference_order()), so a rung
 * of one family is never considered on a machine of another.
 */
enum class Architecture : std::uint8_t {
    Other   = 0, ///< Neither x86 nor aarch64: only the Baseline rung exists.
    X86     = 1, ///< x86 / x86-64: the psABI ladder Baseline, V2, V3, V4.
    Aarch64 = 2, ///< aarch64: Baseline (NEON) and the optional Sme rung.
};

/**
 * @brief CPU features detected at runtime.
 *
 * This is the runtime counterpart of the compile-time `has_*` constants in
 * Platform.hpp: the same feature vocabulary, but describing the machine the
 * process is actually running on rather than the ISA baseline the translation
 * unit was compiled for. Field names mirror the `has_*` constants.
 *
 * All x86 vector-extension fields report *usability*, not just CPU support:
 * a feature is only reported when the CPU advertises it via CPUID *and* the
 * operating system has enabled the corresponding register state (OSXSAVE +
 * XCR0 bits, checked with `xgetbv`). A CPU with AVX2 silicon under an OS
 * that never enabled YMM state reports `avx2 == false`, because issuing an
 * AVX2 instruction there faults. This is the gate most hand-rolled
 * detectors forget.
 *
 * On non-x86, non-aarch64 targets every field is false; such targets run the
 * baseline (scalar) code path.
 */
struct CpuFeatures {
    /// The family of the machine; detect() sets it, and it picks the ladder the other fields are read against.
    Architecture arch = Architecture::Other;

    // ---- x86 ----
    bool sse2  = false; ///< SSE2 (part of the x86-64 baseline; true on every x86-64 CPU).
    bool sse3  = false; ///< SSE3.
    bool ssse3 = false; ///< Supplemental SSE3.
    bool sse41 = false; ///< SSE4.1.
    bool sse42 = false; ///< SSE4.2.

    bool popcnt     = false; ///< POPCNT instruction.
    bool cmpxchg16b = false; ///< CMPXCHG16B instruction.
    bool lahf_sahf  = false; ///< LAHF/SAHF in 64-bit mode.
    bool bmi1       = false; ///< Bit-manipulation instructions 1.
    bool bmi2       = false; ///< Bit-manipulation instructions 2.
    bool f16c       = false; ///< FP16 <-> FP32 conversion instructions.
    bool lzcnt      = false; ///< LZCNT instruction.
    bool movbe      = false; ///< MOVBE instruction.

    bool avx  = false; ///< AVX, gated on OS YMM state (see class docs).
    bool avx2 = false; ///< AVX2, gated on OS YMM state.
    bool fma  = false; ///< FMA3, gated on OS YMM state.

    bool avx512f  = false; ///< AVX-512 Foundation, gated on OS ZMM state.
    bool avx512vl = false; ///< AVX-512 Vector Length extensions, gated on OS ZMM state.
    bool avx512bw = false; ///< AVX-512 Byte/Word, gated on OS ZMM state.
    bool avx512dq = false; ///< AVX-512 Doubleword/Quadword, gated on OS ZMM state.
    bool avx512cd = false; ///< AVX-512 Conflict Detection, gated on OS ZMM state.

    bool avx512_fp16 = false; ///< AVX-512 FP16 arithmetic, gated on OS ZMM state.
    bool avx512_bf16 = false; ///< AVX-512 BF16 arithmetic, gated on OS ZMM state.
    bool avx_vnni    = false; ///< 256-bit VNNI (Alder Lake+), gated on OS YMM state.
    bool avx512_vnni = false; ///< 512-bit VNNI, gated on OS ZMM state.

    bool osxsave   = false; ///< OS advertises XSAVE/XRSTOR support (CPUID.1:ECX.OSXSAVE).
    bool os_avx    = false; ///< OS enabled XMM+YMM state in XCR0; prerequisite for all AVX-family reports.
    bool os_avx512 = false; ///< OS enabled opmask+ZMM state in XCR0; prerequisite for all AVX-512 reports.

    // ---- ARM ----
    bool neon         = false; ///< Advanced SIMD (baseline on aarch64; true on every aarch64 CPU).
    bool neon_fp16    = false; ///< FEAT_FP16: native half-precision vector arithmetic.
    bool neon_bf16    = false; ///< FEAT_BF16: bfloat16 vector instructions.
    bool neon_i8mm    = false; ///< FEAT_I8MM: int8 matrix-multiply instructions.
    bool neon_dotprod = false; ///< FEAT_DotProd: vdotq int8 dot product.

    bool sve  = false; ///< FEAT_SVE: non-streaming Scalable Vector Extension.
    bool sve2 = false; ///< FEAT_SVE2: non-streaming SVE2.
    /// The SVE vector length this thread runs at, in bits (0 without SVE). The sve<N> rungs are
    /// compiled for one length and run only where it is exactly that.
    int sve_vector_bits = 0;

    bool sme        = false; ///< FEAT_SME: Scalable Matrix Extension (streaming SVE + ZA tiles).
    bool sme2       = false; ///< FEAT_SME2: SME2 (multi-vector, required by the sme rung).
    bool sme_f64f64 = false; ///< FEAT_SME_F64F64: FP64 outer-product FMOPA into ZA64 tiles.
};

/**
 * @brief Detect the features of the CPU this process is running on.
 *
 * The detection runs once (thread-safe, on first call) and the result is
 * cached for the lifetime of the process.
 *
 * @return A reference to the cached feature set.
 */
STRIPES_EXPORT CpuFeatures const &cpu_features();

/**
 * @brief The rungs of the runtime dispatch ladder.
 *
 * The x86 rungs follow the psABI micro-architecture levels
 * (`x86-64-v2/-v3/-v4`), which are the industry-standard grouping of ISA
 * extensions and map one-to-one onto compiler flags (`-march=x86-64-v3`,
 * MSVC `/arch:AVX2`, ...):
 *
 * - `Baseline`: what the toolchain's default target provides. SSE2 on
 *   x86-64, NEON on aarch64. Always runnable, by construction.
 * - `V2` (x86-64-v2): SSE3/SSSE3/SSE4.1/SSE4.2, POPCNT, CMPXCHG16B
 *   (Nehalem 2008 / Jaguar 2013 and newer).
 * - `V3` (x86-64-v3): adds AVX/AVX2, FMA, BMI1/BMI2, F16C, LZCNT, MOVBE
 *   (Haswell 2013 / Excavator 2015 and newer).
 * - `V4` (x86-64-v4): adds AVX-512 F/BW/CD/DQ/VL
 *   (Skylake-X 2017 / Zen 4 2022 and newer).
 *
 * On aarch64 `Baseline` is NEON, and `Sme` is the one optional rung: SME2
 * with FP64 outer products (FEAT_SME2 + FEAT_SME_F64F64, Apple M4 and
 * newer). Runtime rungs for further aarch64 features (FEAT_FP16,
 * FEAT_BF16) are future extensions of this enum.
 *
 * The enumerator values are identifiers, not a ranking. Whether a machine
 * can run a rung is supports(), and the order in which rungs are preferred
 * is preference_order(), which is per architecture: aarch64 features do not
 * nest the way the x86 psABI levels do (Apple M4 has SME without SVE), so
 * no single ordinal can say which rung "implies" another.
 */
enum class InstructionSet : std::uint8_t {
    Baseline = 0, ///< Toolchain default target: SSE2 on x86-64, NEON on aarch64.
    V2       = 1, ///< x86-64-v2 (SSE4.2 era).
    V3       = 2, ///< x86-64-v3 (AVX2 + FMA era).
    V4       = 3, ///< x86-64-v4 (AVX-512 era).
    Sme      = 4, ///< aarch64 SME2 + FP64 FMOPA (Apple M4 era).
    Sve128   = 5, ///< aarch64 SVE, compiled for a vector length of exactly 128 bits (Graviton4, Grace).
    Sve256   = 6, ///< aarch64 SVE, compiled for exactly 256 bits (Graviton3).
    Sve512   = 7, ///< aarch64 SVE, compiled for exactly 512 bits (A64FX).
};

/**
 * @brief Human-readable name of a rung: "baseline", "x86-64-v2", ...
 *
 * @param[in] set The rung to name.
 *
 * @return A static string; never nullptr.
 */
STRIPES_EXPORT char const *to_string(InstructionSet set) noexcept;

/**
 * @brief Width in bits of the vector register a rung's kernels are written for.
 *
 * Baseline and V2 are SSE2/SSE4.2 (128), V3 is AVX2 (256), V4 is AVX-512 (512).
 * The aarch64 rungs report NEON's 128: the SME rung's ZA tiles are a matrix
 * unit with its own geometry (see PackedGemm's SME kernel), not a wider
 * vector, and its NEON-side code is still 128-bit.
 *
 * This is the one place the rung-to-width mapping lives; hardware::cpu_info()
 * derives `simd_width_f64` from it so that blocking built from that field
 * agrees with the kernel ladder's choice.
 *
 * @param[in] set The rung.
 *
 * @return 128, 256 or 512.
 */
STRIPES_EXPORT int vector_bits(InstructionSet set) noexcept;

/**
 * @brief Parse a rung name, as accepted by the `STRIPES_ARCH`
 *        option.
 *
 * Accepted spellings (case-insensitive): `baseline`, `v2`, `v3`, `v4`,
 * `x86-64-v2/-v3/-v4`, `sme`, and the colloquial aliases `sse2` (baseline),
 * `sse4.2` (v2), `avx2` (v3), `avx512` (v4), `sme2` (sme).
 *
 * @param[in] name The spelling to parse.
 *
 * @return The rung, or std::nullopt if the spelling is not recognized.
 */
STRIPES_EXPORT std::optional<InstructionSet> parse_instruction_set(std::string_view name);

/**
 * @brief The rungs of one architecture's ladder, most preferred first.
 *
 * x86 is `{V4, V3, V2, Baseline}` and aarch64 is `{Sme, Baseline}`; every
 * other architecture has only `{Baseline}`. Every list ends in Baseline.
 * Dispatch walks this list, never the enumerator values.
 *
 * @param[in] arch The architecture whose ladder to return.
 *
 * @return A view of a static array; valid for the lifetime of the process.
 */
STRIPES_EXPORT std::span<InstructionSet const> preference_order(Architecture arch) noexcept;

/**
 * @brief Whether a machine with @p features can execute code compiled for @p set.
 *
 * Pure function of the feature set. A rung of another architecture is never
 * supported. The x86 levels apply the full psABI gate (every listed
 * extension, including the OS-state gates folded into CpuFeatures). The
 * `Sme` rung needs SME2 with FP64 outer products, and also whatever else the
 * compiler enables for the rung's translation units: a compiler that turns
 * on non-streaming SVE or SVE2 with `+sme2` (GCC before 15) may emit it
 * anywhere in those TUs, so for such a build the rung needs SVE or SVE2 too.
 *
 * @param[in] features The feature set to test.
 * @param[in] set The rung.
 *
 * @return True when every feature the rung's code may use is present.
 */
STRIPES_EXPORT bool supports(CpuFeatures const &features, InstructionSet set) noexcept;

/**
 * @brief The most preferred rung this CPU can execute.
 *
 * The first entry of `preference_order(features.arch)` that supports()
 * accepts. Useful directly in tests; most callers want selected_arch()
 * instead.
 *
 * @param[in] features The feature set to classify.
 *
 * @return The best supported rung; Baseline when nothing else qualifies.
 */
STRIPES_EXPORT InstructionSet highest_supported(CpuFeatures const &features) noexcept;

/**
 * @brief Resolve the rung to dispatch to, given a feature set and an
 *        optional override spelling.
 *
 * The override (set_arch_override() or the `STRIPES_ARCH` environment variable) can
 * only choose a rung the machine supports. A supported rung is used as
 * given. A rung of this architecture that the machine cannot run is
 * replaced, with a warning, by the next supported rung after it in
 * preference_order(), so asking for `v4` on an AVX2 machine gives `v3`. A
 * rung of another architecture, or an unparseable spelling, is ignored with
 * a logged warning. This is the pure, deterministic core of selected_arch(),
 * separated so tests can drive it with synthetic feature sets and override
 * strings.
 *
 * @param[in] features The detected (or synthetic) feature set.
 * @param[in] override_name Optional rung spelling; pass std::nullopt for "no override".
 *
 * @return The rung to dispatch to.
 */
STRIPES_EXPORT InstructionSet resolve_arch(CpuFeatures const &features, std::optional<std::string_view> override_name);

/**
 * @brief The rung the current process dispatches to.
 *
 * Equivalent to `resolve_arch(cpu_features(), override)`, computed once
 * (thread-safe) and cached for the lifetime of the process. The override is
 * the last set_arch_override() made before the first call, or failing that
 * the `STRIPES_ARCH` environment variable. Changing either afterwards has no
 * effect. Test code that needs different rungs should call resolve_arch()
 * directly.
 *
 * @return The cached rung.
 */
STRIPES_EXPORT InstructionSet selected_arch();

/**
 * @brief Cap the rung selected_arch() will choose, as `STRIPES_ARCH` does.
 *
 * For a program or library with its own configuration (a command-line flag,
 * a settings file) to pass the user's choice on; it takes precedence over the
 * environment variable. An empty name removes the override. It must be made
 * before the first selected_arch() call in the process, since that call
 * caches its answer.
 *
 * @return false when selected_arch() has already chosen, so the override came
 *         too late to take effect.
 */
STRIPES_EXPORT bool set_arch_override(std::string_view name);

/// How serious a message from the runtime library is.
enum class MessageLevel { Debug, Warning };

/// Receives the runtime library's messages: warnings about an override that
/// cannot be honored, and the rung chosen, at debug level.
using MessageHandler = void (*)(MessageLevel level, std::string_view message);

/**
 * @brief Route the runtime library's messages, for a program that has its own
 *        logging. The default writes warnings to stderr and drops debug
 *        messages; passing nullptr restores it.
 */
STRIPES_EXPORT void set_message_handler(MessageHandler handler) noexcept;

/**
 * @brief What the code compiled for the sme rung needs besides SME itself.
 *
 * supports() says whether a core has the rung's own instructions (SME2 and
 * FP64 FMOPA). Some compilers' SME flags also switch on non-streaming SVE or
 * SVE2 (GCC before 15 makes +sme imply +sve2), and the autovectorizer may then
 * put either anywhere in the rung's translation units, which faults on a core
 * with SME but no SVE (Apple M4). Whether that happened depends on the
 * compiler that built the calling code, not on the one that built Stripes, so
 * the caller passes it: stripes_add_dispatch_sources() probes it and defines
 * STRIPES_SME_RUNG_ENABLES_SVE and _SVE2 in its definitions, and
 * STRIPES_LADDER() hands them to select(). The default assumes the worst.
 */
struct SmeRungRequires {
    bool sve  = true;
    bool sve2 = true;
};

/// Whether code built for the sme rung with these requirements runs on @p features.
inline bool sme_rung_runs(CpuFeatures const &features, SmeRungRequires requires_) noexcept {
    return supports(features, InstructionSet::Sme) && (!requires_.sve || features.sve) && (!requires_.sve2 || features.sve2);
}

/**
 * @brief One kernel's entry point for each rung, as select() takes them.
 *
 * A rung that was not built is nullptr; `baseline` must always be set. STRIPES_LADDER(fn) fills
 * one from the copies stripes_add_dispatch_sources() built, so a call site does not change when
 * a rung is added. Written by hand, designated initializers name only the rungs there are:
 *
 * @code
 * static KernelFn const kernel = stripes::select<KernelFn>({.baseline = &arch_baseline::kernel, .v3 = &arch_v3::kernel});
 * @endcode
 */
template <typename F>
struct Ladder {
    F               baseline = nullptr;
    F               v2       = nullptr;
    F               v3       = nullptr;
    F               v4       = nullptr;
    F               sme      = nullptr;
    F               sve128   = nullptr;
    F               sve256   = nullptr;
    F               sve512   = nullptr;
    SmeRungRequires sme_requires{}; ///< What the sme entry's code needs besides SME; the default assumes the worst.

    /// The entry for @p set, nullptr if it was not built.
    constexpr F entry(InstructionSet set) const noexcept {
        switch (set) {
        case InstructionSet::Baseline:
            return baseline;
        case InstructionSet::V2:
            return v2;
        case InstructionSet::V3:
            return v3;
        case InstructionSet::V4:
            return v4;
        case InstructionSet::Sme:
            return sme;
        case InstructionSet::Sve128:
            return sve128;
        case InstructionSet::Sve256:
            return sve256;
        case InstructionSet::Sve512:
            return sve512;
        }
        return nullptr;
    }

    /// Whether @p set's entry was built and runs on @p features (with the sme entry's requirements).
    bool runs(CpuFeatures const &features, InstructionSet set) const noexcept {
        if (entry(set) == nullptr) {
            return false;
        }
        return set == InstructionSet::Sme ? sme_rung_runs(features, sme_requires) : supports(features, set);
    }
};

/**
 * @brief select() against an explicit feature set and starting rung.
 *
 * Walks `preference_order(features.arch)` from @p start onward and returns the first entry that
 * was built and runs here (Ladder::runs). The test matters on aarch64, where a rung later in the
 * list is not implied by an earlier one. select() calls this with cpu_features() and
 * selected_arch(); tests call it directly with synthetic machines.
 *
 * @return The entry point to call; never nullptr.
 */
template <typename F>
F select_for(CpuFeatures const &features, InstructionSet start, Ladder<F> const &ladder) {
    bool reached = false;
    for (InstructionSet const rung : preference_order(features.arch)) {
        reached = reached || rung == start;
        if (reached && ladder.runs(features, rung)) {
            return ladder.entry(rung);
        }
    }
    return ladder.baseline;
}

/**
 * @brief Pick the best available entry point for the selected rung.
 *
 * Generic dispatch helper for modules that compile a kernel once per rung: given each rung's
 * entry point (nullptr for rungs not built), it returns the entry for the most preferred built
 * rung at or after selected_arch() in preference_order() that runs on this machine. Resolve it
 * once and keep it; the choice cannot change while the program runs.
 *
 * @code
 * static KernelFn const kernel = stripes::select<KernelFn>(STRIPES_LADDER(kernel));
 * @endcode
 *
 * @return The entry point to call; never nullptr.
 */
template <typename F>
F select(Ladder<F> const &ladder) {
    return select_for<F>(cpu_features(), selected_arch(), ladder);
}

STRIPES_NAMESPACE_END()
