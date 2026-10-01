//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// selected_arch() chooses once per process, so each case here runs as its own process (one ctest
// registration per tag, see CMakeLists.txt) with the environment that registration gives it.

#include <Stripes/RuntimeFeatures.hpp>
#include <cstdlib>
#include <string>
#include <vector>

#include <catch2/catch_all.hpp>

using stripes::InstructionSet;

namespace {
std::vector<std::string> warnings;
void                     capture(stripes::MessageLevel level, std::string_view message) {
    if (level == stripes::MessageLevel::Warning) {
        warnings.emplace_back(message);
    }
}
} // namespace

TEST_CASE("set_arch_override before the first selection decides it", "[override]") {
    REQUIRE(stripes::set_arch_override("baseline"));
    CHECK(stripes::selected_arch() == InstructionSet::Baseline);
    // Too late now: the choice is cached, and the call says so.
    CHECK_FALSE(stripes::set_arch_override("v2"));
    CHECK(stripes::selected_arch() == InstructionSet::Baseline);
}

TEST_CASE("STRIPES_ARCH caps the selection", "[env]") {
    // Registered with STRIPES_ARCH=baseline.
    REQUIRE(std::getenv("STRIPES_ARCH") != nullptr);
    CHECK(stripes::selected_arch() == InstructionSet::Baseline);
}

TEST_CASE("set_arch_override wins over STRIPES_ARCH", "[override-env]") {
    // Registered with STRIPES_ARCH=bogus, which would warn if it were read.
    stripes::set_message_handler(&capture);
    REQUIRE(stripes::set_arch_override("baseline"));
    CHECK(stripes::selected_arch() == InstructionSet::Baseline);
    CHECK(warnings.empty());
    stripes::set_message_handler(nullptr);
}

TEST_CASE("an empty override hands the choice back to STRIPES_ARCH", "[override-cleared]") {
    // Registered with STRIPES_ARCH=baseline. A program forwarding a setting of its own that the user
    // left empty passes "", which must not shadow the environment.
    REQUIRE(stripes::set_arch_override("v2"));
    REQUIRE(stripes::set_arch_override(""));
    CHECK(stripes::selected_arch() == InstructionSet::Baseline);
}

TEST_CASE("an unusable STRIPES_ARCH reaches the message handler", "[handler]") {
    stripes::set_message_handler(&capture);
    stripes::CpuFeatures const host = stripes::cpu_features();
    CHECK(stripes::resolve_arch(host, std::string_view("not-a-rung")) == stripes::highest_supported(host));
    REQUIRE(warnings.size() == 1);
    CHECK(warnings[0].find("not-a-rung") != std::string::npos);

    // nullptr restores the default (stderr), which no longer reaches capture().
    stripes::set_message_handler(nullptr);
    (void)stripes::resolve_arch(host, std::string_view("not-a-rung"));
    CHECK(warnings.size() == 1);
}
