//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Test launcher for per-rung SIMD dispatch tests (stripes_add_rung_tests).
//
// Usage: stripes_rung_guard [--require-sve] [--require-sve2] <rung> <command> [args...]
//
// The --require options are what the tested code's sme copies need besides
// SME (stripes::SmeRungRequires), which the build probed with the compiler
// that built them; select() would quietly fall back to baseline without them,
// so the guard skips instead.
//
// If the host CPU cannot execute <rung>, exits with 77 - registered as the
// test's SKIP_RETURN_CODE - so ctest reports an honest "Skipped" instead of
// silently rerunning another rung (STRIPES_ARCH replaces an unsupported
// rung with a supported one). Otherwise replaces itself with <command>.

#include <Stripes/RuntimeFeatures.hpp>
#include <cstdio>
#include <string_view>

#if defined(_WIN32)
#    include <process.h>
#else
#    include <unistd.h>
#endif

int main(int argc, char **argv) {
    stripes::SmeRungRequires sme_requires{false, false};
    int                      first = 1;
    for (; first < argc && argv[first][0] == '-' && argv[first][1] == '-'; ++first) {
        std::string_view const option = argv[first];
        if (option == "--require-sve") {
            sme_requires.sve = true;
        } else if (option == "--require-sve2") {
            sme_requires.sve2 = true;
        } else {
            std::fprintf(stderr, "stripes_rung_guard: unknown option '%s'\n", argv[first]);
            return 2;
        }
    }
    if (argc - first < 2) {
        std::fprintf(stderr, "usage: stripes_rung_guard [--require-sve] [--require-sve2] <rung> <command> [args...]\n");
        return 2;
    }

    auto const requested = stripes::parse_instruction_set(argv[first]);
    if (!requested.has_value()) {
        std::fprintf(stderr, "stripes_rung_guard: unknown rung '%s'\n", argv[first]);
        return 2;
    }

    auto const &features = stripes::cpu_features();
    bool const  runs     = *requested == stripes::InstructionSet::Sme ? stripes::sme_rung_runs(features, sme_requires)
                                                                      : stripes::supports(features, *requested);
    if (!runs) {
        std::fprintf(stderr, "stripes_rung_guard: host cannot run %s (best supported: %s); skipping the test\n",
                     stripes::to_string(*requested), stripes::to_string(stripes::highest_supported(features)));
        return 77; // SKIP_RETURN_CODE
    }

    char **const command = argv + first + 1;
#if defined(_WIN32)
    auto const rc = _execv(command[0], command);
#else
    auto const rc = execv(command[0], command);
#endif
    std::fprintf(stderr, "stripes_rung_guard: failed to exec '%s' (rc=%d)\n", command[0], static_cast<int>(rc));
    return 2;
}
