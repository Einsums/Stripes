//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Stripes/Platform.hpp>
#include <Stripes/RungLadder.hpp>
#include <Stripes/RuntimeFeatures.hpp>

#include <catch2/catch_all.hpp>

using stripes::CpuFeatures;
using stripes::InstructionSet;

namespace {

// Synthetic feature sets for the pure ladder functions. Field-by-field
// construction mirrors what detect() produces on real hardware.
CpuFeatures features_v2() {
    CpuFeatures f;
    f.arch       = stripes::Architecture::X86;
    f.sse2       = true;
    f.sse3       = true;
    f.ssse3      = true;
    f.sse41      = true;
    f.sse42      = true;
    f.popcnt     = true;
    f.cmpxchg16b = true;
    f.lahf_sahf  = true;
    return f;
}

CpuFeatures features_v3() {
    CpuFeatures f = features_v2();
    f.osxsave     = true;
    f.os_avx      = true;
    f.avx         = true;
    f.avx2        = true;
    f.fma         = true;
    f.bmi1        = true;
    f.bmi2        = true;
    f.f16c        = true;
    f.lzcnt       = true;
    f.movbe       = true;
    return f;
}

CpuFeatures features_v4() {
    CpuFeatures f = features_v3();
    f.os_avx512   = true;
    f.avx512f     = true;
    f.avx512vl    = true;
    f.avx512bw    = true;
    f.avx512dq    = true;
    f.avx512cd    = true;
    return f;
}

// An aarch64 core with SME2 and FP64 outer products but no non-streaming
// SVE: Apple M4.
CpuFeatures features_m4() {
    CpuFeatures f;
    f.arch       = stripes::Architecture::Aarch64;
    f.neon       = true;
    f.sme        = true;
    f.sme2       = true;
    f.sme_f64f64 = true;
    return f;
}

// An Armv9 core with SVE2 and the same SME features.
CpuFeatures features_sme_with_sve2() {
    CpuFeatures f = features_m4();
    f.sve         = true;
    f.sve2        = true;
    return f;
}

// An Armv9 core with SVE2 and no SME (Graviton4, Grace).
CpuFeatures features_sve2_only() {
    CpuFeatures f;
    f.arch            = stripes::Architecture::Aarch64;
    f.neon            = true;
    f.sve             = true;
    f.sve2            = true;
    f.sve_vector_bits = 128;
    return f;
}

// What the sme rung's flags enable on this compiler, probed at configure time.
constexpr bool sme_rung_enables_sve  = STRIPES_SME_RUNG_ENABLES_SVE != 0;
constexpr bool sme_rung_enables_sve2 = STRIPES_SME_RUNG_ENABLES_SVE2 != 0;

} // namespace

TEST_CASE("cpu_features: sane on the host", "[simd][runtime-features]") {
    auto const &f = stripes::cpu_features();

#if defined(__x86_64__) || defined(_M_X64)
    // SSE2 is part of the x86-64 baseline; every x86-64 CPU has it.
    REQUIRE(f.arch == stripes::Architecture::X86);
    REQUIRE(f.sse2);
    REQUIRE_FALSE(f.neon);
#elif defined(__aarch64__) || defined(_M_ARM64)
    REQUIRE(f.arch == stripes::Architecture::Aarch64);
    REQUIRE(f.neon);
    REQUIRE_FALSE(f.sse2);
    if (f.sve2) {
        CHECK(f.sve);
    }
#endif

    // Feature-implication sanity: these hold on all real silicon.
    if (f.sse42) {
        CHECK(f.sse41);
        CHECK(f.ssse3);
        CHECK(f.sse2);
    }
    if (f.avx2) {
        CHECK(f.avx);
        CHECK(f.os_avx);
    }
    if (f.avx512f) {
        CHECK(f.os_avx512);
        CHECK(f.avx2);
    }

    // Detection is cached: same object every call.
    CHECK(&stripes::cpu_features() == &f);
}

TEST_CASE("cpu_features: agrees with the compile-time baseline", "[simd][runtime-features]") {
    auto const &f = stripes::cpu_features();

    // Whatever ISA this test was COMPILED for must be present at runtime,
    // or the binary could not be running. This cross-checks detect()
    // against Platform.hpp on every machine the test suite runs on.
    if constexpr (stripes::has_sse42) {
        CHECK(f.sse42);
    }
    if constexpr (stripes::has_avx2) {
        CHECK(f.avx2);
    }
    if constexpr (stripes::has_avx512) {
        CHECK(f.avx512f);
        CHECK(f.avx512vl);
    }
    if constexpr (stripes::has_neon) {
        CHECK(f.neon);
    }
}

TEST_CASE("highest_supported: full psABI gates", "[simd][runtime-features]") {
    CHECK(stripes::highest_supported(CpuFeatures{}) == InstructionSet::Baseline);
    CHECK(stripes::highest_supported(features_v2()) == InstructionSet::V2);
    CHECK(stripes::highest_supported(features_v3()) == InstructionSet::V3);
    CHECK(stripes::highest_supported(features_v4()) == InstructionSet::V4);

    SECTION("one missing extension drops the whole level") {
        auto f = features_v3();
        f.bmi2 = false;
        CHECK(stripes::highest_supported(f) == InstructionSet::V2);
    }

    SECTION("CPU support without OS state does not qualify") {
        auto f      = features_v4();
        f.os_avx512 = false;
        CHECK(stripes::highest_supported(f) == InstructionSet::V3);

        auto g   = features_v3();
        g.os_avx = false;
        CHECK(stripes::highest_supported(g) == InstructionSet::V2);
    }
}

TEST_CASE("parse_instruction_set / to_string", "[simd][runtime-features]") {
    using stripes::parse_instruction_set;

    CHECK(parse_instruction_set("baseline") == InstructionSet::Baseline);
    CHECK(parse_instruction_set("v2") == InstructionSet::V2);
    CHECK(parse_instruction_set("v3") == InstructionSet::V3);
    CHECK(parse_instruction_set("v4") == InstructionSet::V4);
    CHECK(parse_instruction_set("x86-64-v3") == InstructionSet::V3);
    CHECK(parse_instruction_set("AVX2") == InstructionSet::V3);
    CHECK(parse_instruction_set("avx512") == InstructionSet::V4);
    CHECK(parse_instruction_set("sse4.2") == InstructionSet::V2);
    CHECK(parse_instruction_set("sse2") == InstructionSet::Baseline);
    CHECK(parse_instruction_set("sme") == InstructionSet::Sme);
    CHECK(parse_instruction_set("SME2") == InstructionSet::Sme);
    CHECK(parse_instruction_set("sve128") == InstructionSet::Sve128);
    CHECK(parse_instruction_set("SVE256") == InstructionSet::Sve256);
    CHECK(parse_instruction_set("sve512") == InstructionSet::Sve512);
    CHECK_FALSE(parse_instruction_set("sve").has_value()); // a length is part of the rung
    CHECK_FALSE(parse_instruction_set("pentium3").has_value());
    CHECK_FALSE(parse_instruction_set("").has_value());

    // Round trip through the canonical names.
    for (auto set : {InstructionSet::Baseline, InstructionSet::V2, InstructionSet::V3, InstructionSet::V4, InstructionSet::Sme,
                     InstructionSet::Sve128, InstructionSet::Sve256, InstructionSet::Sve512}) {
        CHECK(parse_instruction_set(stripes::to_string(set)) == set);
    }
}

TEST_CASE("preference_order: one ladder per architecture, ending in Baseline", "[simd][runtime-features]") {
    using stripes::Architecture;
    using stripes::preference_order;

    auto const x86 = preference_order(Architecture::X86);
    REQUIRE(x86.size() == 4);
    CHECK(x86[0] == InstructionSet::V4);
    CHECK(x86[1] == InstructionSet::V3);
    CHECK(x86[2] == InstructionSet::V2);
    CHECK(x86[3] == InstructionSet::Baseline);

    auto const arm = preference_order(Architecture::Aarch64);
    REQUIRE(arm.size() == 5);
    CHECK(arm[0] == InstructionSet::Sme);
    CHECK(arm[1] == InstructionSet::Sve512);
    CHECK(arm[2] == InstructionSet::Sve256);
    CHECK(arm[3] == InstructionSet::Sve128);
    CHECK(arm[4] == InstructionSet::Baseline);

    auto const other = preference_order(Architecture::Other);
    REQUIRE(other.size() == 1);
    CHECK(other[0] == InstructionSet::Baseline);
}

TEST_CASE("supports: rungs never cross architectures", "[simd][runtime-features]") {
    using stripes::supports;

    // Baseline runs everywhere.
    CHECK(supports(CpuFeatures{}, InstructionSet::Baseline));
    CHECK(supports(features_v4(), InstructionSet::Baseline));
    CHECK(supports(features_m4(), InstructionSet::Baseline));

    // An x86 machine never runs the sme rung, and an aarch64 machine never
    // runs an x86 level, whatever the other fields claim.
    CHECK_FALSE(supports(features_v4(), InstructionSet::Sme));
    auto confused = features_v4();
    confused.arch = stripes::Architecture::Aarch64;
    CHECK_FALSE(supports(confused, InstructionSet::V2));
    CHECK_FALSE(supports(confused, InstructionSet::V4));

    CHECK(supports(features_v3(), InstructionSet::V2));
    CHECK(supports(features_v3(), InstructionSet::V3));
    CHECK_FALSE(supports(features_v3(), InstructionSet::V4));
}

TEST_CASE("supports: the sme rung is the hardware's SME2 and FP64 FMOPA", "[simd][runtime-features]") {
    using stripes::supports;

    CHECK(supports(features_sme_with_sve2(), InstructionSet::Sme));
    // Apple M4: the rung's instructions are there; whether a given build's sme
    // code runs there is sme_rung_runs()'s question, not this one.
    CHECK(supports(features_m4(), InstructionSet::Sme));

    // SVE without SME is not enough.
    CHECK_FALSE(supports(features_sve2_only(), InstructionSet::Sme));

    // One missing SME feature drops the rung.
    auto no_f64       = features_sme_with_sve2();
    no_f64.sme_f64f64 = false;
    CHECK_FALSE(supports(no_f64, InstructionSet::Sme));
}

TEST_CASE("sme_rung_runs: the code's requirements on top of the rung", "[simd][runtime-features]") {
    using stripes::sme_rung_runs;
    using stripes::SmeRungRequires;

    // Built by a compiler whose +sme2 stays out of SVE (Clang, GCC 15): M4 runs it.
    CHECK(sme_rung_runs(features_m4(), SmeRungRequires{false, false}));
    // Built by one whose +sme2 also switches on SVE or SVE2 (GCC before 15): the
    // code may hold non-streaming SVE anywhere, which faults on M4.
    CHECK_FALSE(sme_rung_runs(features_m4(), SmeRungRequires{true, false}));
    CHECK_FALSE(sme_rung_runs(features_m4(), SmeRungRequires{false, true}));
    // Unknown requirements assume the worst.
    CHECK_FALSE(sme_rung_runs(features_m4(), SmeRungRequires{}));

    // A core with SVE2 runs it however it was built.
    CHECK(sme_rung_runs(features_sme_with_sve2(), SmeRungRequires{}));
    // No SME, no rung, whatever the requirements.
    CHECK_FALSE(sme_rung_runs(features_sve2_only(), SmeRungRequires{false, false}));
}

TEST_CASE("STRIPES_SME_REQUIRES carries this target's probe", "[simd][runtime-features]") {
    // This target is given the probed STRIPES_SME_RUNG_ENABLES_* (CMakeLists.txt), as
    // stripes_add_dispatch_sources() gives them to a target that builds the sme rung.
    stripes::SmeRungRequires const r = STRIPES_SME_REQUIRES;
    CHECK(r.sve == sme_rung_enables_sve);
    CHECK(r.sve2 == sme_rung_enables_sve2);
}

TEST_CASE("sme rung probe agrees with the known compiler behaviour", "[simd][runtime-features]") {
#if (defined(__aarch64__) || defined(_M_ARM64)) && defined(__GNUC__) && !defined(__clang__)
    // GCC 14's +sme depends on +sve2; GCC 15 dropped that dependency.
    if constexpr (__GNUC__ < 15) {
        CHECK(sme_rung_enables_sve);
        CHECK(sme_rung_enables_sve2);
    } else {
        CHECK_FALSE(sme_rung_enables_sve);
        CHECK_FALSE(sme_rung_enables_sve2);
    }
#elif (defined(__aarch64__) || defined(_M_ARM64)) && defined(__clang__)
    // Clang keeps SME and SVE independent.
    CHECK_FALSE(sme_rung_enables_sve);
    CHECK_FALSE(sme_rung_enables_sve2);
#else
    // No sme rung off aarch64.
    CHECK_FALSE(sme_rung_enables_sve);
    CHECK_FALSE(sme_rung_enables_sve2);
#endif
}

TEST_CASE("highest_supported: aarch64 ladder", "[simd][runtime-features]") {
    CHECK(stripes::highest_supported(features_sme_with_sve2()) == InstructionSet::Sme);
    // SVE2 at 128 bits without SME (Graviton4): the sve128 rung.
    CHECK(stripes::highest_supported(features_sve2_only()) == InstructionSet::Sve128);
    CpuFeatures no_vl     = features_sve2_only();
    no_vl.sve_vector_bits = 0; // length unknown: no sve rung
    CHECK(stripes::highest_supported(no_vl) == InstructionSet::Baseline);
    CHECK(stripes::highest_supported(features_m4()) == InstructionSet::Sme);
}

TEST_CASE("resolve_arch: override semantics", "[simd][runtime-features]") {
    using stripes::resolve_arch;

    auto const v4 = features_v4();

    SECTION("no override selects the hardware ceiling") {
        CHECK(resolve_arch(v4, std::nullopt) == InstructionSet::V4);
        CHECK(resolve_arch(features_v2(), std::nullopt) == InstructionSet::V2);
    }

    SECTION("override can lower the rung") {
        CHECK(resolve_arch(v4, "baseline") == InstructionSet::Baseline);
        CHECK(resolve_arch(v4, "v2") == InstructionSet::V2);
        CHECK(resolve_arch(v4, "avx2") == InstructionSet::V3);
    }

    SECTION("override above the ceiling clamps to the next supported rung") {
        CHECK(resolve_arch(features_v2(), "v4") == InstructionSet::V2);
        CHECK(resolve_arch(features_v3(), "v4") == InstructionSet::V3);
        CHECK(resolve_arch(CpuFeatures{}, "avx2") == InstructionSet::Baseline);
    }

    SECTION("unparseable override is ignored") {
        CHECK(resolve_arch(v4, "fastest-please") == InstructionSet::V4);
    }

    SECTION("a rung of another architecture is ignored") {
        CHECK(resolve_arch(v4, "sme") == InstructionSet::V4);
        CHECK(resolve_arch(features_sme_with_sve2(), "v2") == InstructionSet::Sme);
        CHECK(resolve_arch(features_sve2_only(), "avx512") == InstructionSet::Sve128);
    }

    SECTION("aarch64 overrides") {
        CHECK(resolve_arch(features_sme_with_sve2(), "baseline") == InstructionSet::Baseline);
        CHECK(resolve_arch(features_sme_with_sve2(), "SME2") == InstructionSet::Sme);
        // A core without SME asking for the sme rung gets the next rung down it runs, here sve128;
        // so does one asking for a vector length it does not have.
        CHECK(resolve_arch(features_sve2_only(), "sme") == InstructionSet::Sve128);
        CHECK(resolve_arch(features_sve2_only(), "sve256") == InstructionSet::Sve128);
        CHECK(resolve_arch(features_sve2_only(), "baseline") == InstructionSet::Baseline);
    }
}

TEST_CASE("selected_arch: cached and supported by the host", "[simd][runtime-features]") {
    auto const first = stripes::selected_arch();
    CHECK(stripes::supports(stripes::cpu_features(), first));
    CHECK(stripes::selected_arch() == first);
}

namespace {
using Fn = int (*)();

int ret_baseline() {
    return 0;
}
int ret_v2() {
    return 2;
}
int ret_v3() {
    return 3;
}
int ret_v4() {
    return 4;
}
int ret_sme() {
    return 5;
}
int ret_sve128() {
    return 128;
}
int ret_sve256() {
    return 256;
}
int ret_sve512() {
    return 512;
}
} // namespace

TEST_CASE("supports: an sve rung needs exactly its vector length", "[simd][runtime-features]") {
    using stripes::supports;

    auto at = [](int bits) {
        CpuFeatures f     = features_sve2_only();
        f.sve_vector_bits = bits;
        return f;
    };
    CHECK(supports(at(128), InstructionSet::Sve128));
    CHECK_FALSE(supports(at(128), InstructionSet::Sve256));
    CHECK(supports(at(256), InstructionSet::Sve256));
    // Longer is not enough: the register would hold lanes the code does not know about.
    CHECK_FALSE(supports(at(512), InstructionSet::Sve256));
    CHECK(supports(at(512), InstructionSet::Sve512));
    // No SVE (Apple M4), or SVE on another architecture, runs none of them.
    CHECK_FALSE(supports(features_m4(), InstructionSet::Sve128));
    CpuFeatures confused     = features_v4();
    confused.sve             = true;
    confused.sve_vector_bits = 256;
    CHECK_FALSE(supports(confused, InstructionSet::Sve256));
    CHECK(stripes::vector_bits(InstructionSet::Sve512) == 512);
}

TEST_CASE("select_for: walks the architecture's order from the start rung", "[simd][runtime-features]") {
    using stripes::select_for;
    using L = stripes::Ladder<Fn>;
    L const all{.baseline = ret_baseline, .v2 = ret_v2, .v3 = ret_v3, .v4 = ret_v4, .sme = ret_sme};

    SECTION("x86: falls through missing rungs toward Baseline") {
        auto const v4 = features_v4();
        CHECK(select_for<Fn>(v4, InstructionSet::V4, all)() == 4);
        CHECK(select_for<Fn>(v4, InstructionSet::V4, L{.baseline = ret_baseline, .v2 = ret_v2, .v3 = ret_v3})() == 3);
        CHECK(select_for<Fn>(v4, InstructionSet::V3, all)() == 3);
        CHECK(select_for<Fn>(v4, InstructionSet::V4, L{.baseline = ret_baseline})() == 0);
        CHECK(select_for<Fn>(v4, InstructionSet::Baseline, all)() == 0);
    }

    SECTION("x86: an sme entry is never taken") {
        CHECK(select_for<Fn>(features_v2(), InstructionSet::V2, L{.baseline = ret_baseline, .sme = ret_sme})() == 0);
    }

    SECTION("aarch64: x86 entries are never taken") {
        // An sme host whose ladder has no sme entry drops to Baseline, not to
        // an x86 slot that happens to be filled.
        auto const sme = features_sme_with_sve2();
        CHECK(select_for<Fn>(sme, InstructionSet::Sme, L{.baseline = ret_baseline, .v2 = ret_v2, .v3 = ret_v3, .v4 = ret_v4})() == 0);
        CHECK(select_for<Fn>(sme, InstructionSet::Sme, all)() == 5);
    }

    SECTION("aarch64: an sme entry whose code needs SVE is skipped on a core without it") {
        using stripes::SmeRungRequires;
        auto const m4 = features_m4();
        CHECK(select_for<Fn>(m4, InstructionSet::Sme,
                             L{.baseline = ret_baseline, .sme = ret_sme, .sme_requires = SmeRungRequires{false, false}})() == 5);
        CHECK(select_for<Fn>(m4, InstructionSet::Sme,
                             L{.baseline = ret_baseline, .sme = ret_sme, .sme_requires = SmeRungRequires{true, true}})() == 0);
        // Without requirements, the worst case.
        CHECK(select_for<Fn>(m4, InstructionSet::Sme, L{.baseline = ret_baseline, .sme = ret_sme})() == 0);
    }

    SECTION("aarch64: an unsupported entry is skipped even from the top") {
        CHECK(select_for<Fn>(features_sve2_only(), InstructionSet::Sme, L{.baseline = ret_baseline, .sme = ret_sme})() == 0);
    }

    SECTION("aarch64: the sve copy of the machine's vector length, and no other") {
        L const sve{.baseline = ret_baseline, .sve128 = ret_sve128, .sve256 = ret_sve256, .sve512 = ret_sve512};
        for (int bits : {128, 256, 512}) {
            CpuFeatures f     = features_sve2_only();
            f.sve_vector_bits = bits;
            CHECK(select_for<Fn>(f, stripes::highest_supported(f), sve)() == bits);
        }
        // A length that was not built runs the NEON baseline.
        CpuFeatures f     = features_sve2_only();
        f.sve_vector_bits = 256;
        CHECK(select_for<Fn>(f, InstructionSet::Sve256, L{.baseline = ret_baseline, .sve128 = ret_sve128})() == 0);
    }

    SECTION("a start rung of another architecture yields Baseline") {
        CHECK(select_for<Fn>(features_v4(), InstructionSet::Sme, all)() == 0);
    }
}

TEST_CASE("select: dispatches the host's selected rung", "[simd][runtime-features]") {
    int const expected = [] {
        switch (stripes::selected_arch()) {
        case InstructionSet::V2:
            return 2;
        case InstructionSet::V3:
            return 3;
        case InstructionSet::V4:
            return 4;
        case InstructionSet::Sme:
            // select() without requirements assumes the sme code needs SVE and SVE2.
            return stripes::sme_rung_runs(stripes::cpu_features(), {}) ? 5 : 0;
        case InstructionSet::Sve128:
            return 128;
        case InstructionSet::Sve256:
            return 256;
        case InstructionSet::Sve512:
            return 512;
        case InstructionSet::Baseline:
            break;
        }
        return 0;
    }();

    CHECK(stripes::select<Fn>({.baseline = ret_baseline,
                               .v2       = ret_v2,
                               .v3       = ret_v3,
                               .v4       = ret_v4,
                               .sme      = ret_sme,
                               .sve128   = ret_sve128,
                               .sve256   = ret_sve256,
                               .sve512   = ret_sve512})() == expected);
    CHECK(stripes::select<Fn>({.baseline = ret_baseline})() == 0);
}
