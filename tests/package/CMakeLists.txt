#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# A project that uses an installed Stripes the way a library does: find_package, a kernel compiled
# once per rung, and its rung tests. CI configures it against a fresh install
# (-DCMAKE_PREFIX_PATH=<prefix>); it is not part of the Stripes build.
cmake_minimum_required(VERSION 3.24)
project(StripesPackageTest LANGUAGES CXX)

find_package(Stripes 1 REQUIRED)

stripes_add_dispatch_sources(KernelRungs IMPL Kernel.cpp)
add_executable(package_test main.cpp ${KernelRungs})
target_link_libraries(package_test PRIVATE Stripes::stripes)
target_compile_definitions(package_test PRIVATE ${KernelRungs_DEFINITIONS})

enable_testing()
add_test(NAME package COMMAND package_test)
stripes_add_rung_tests(TARGET package_test NAME package)
