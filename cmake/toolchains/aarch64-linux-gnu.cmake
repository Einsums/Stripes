#----------------------------------------------------------------------------------------------
# Copyright (c) The Einsums Developers. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
#----------------------------------------------------------------------------------------------

# Cross-compile for aarch64 Linux with the Debian/Ubuntu GNU cross toolchain
# (g++-aarch64-linux-gnu). The tests run under QEMU user mode: binfmt_misc
# hands every aarch64 binary to qemu-aarch64, so the rung guard's exec of a
# test works too. Set QEMU_CPU=max for SVE, SVE2, SME and SME2, and
# QEMU_LD_PREFIX=/usr/aarch64-linux-gnu for the target's libraries.
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(STRIPES_CROSS_GCC_VERSION
    ""
    CACHE STRING "Suffix of the cross compiler, such as -14 for aarch64-linux-gnu-g++-14"
)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++${STRIPES_CROSS_GCC_VERSION})

set(CMAKE_FIND_ROOT_PATH /usr/aarch64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
