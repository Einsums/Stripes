//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Must not compile: a CUDA translation unit has only the scalar half of Stripes, and Vec.hpp says so
// with an #error instead of failing somewhere in the intrinsic headers.

#include <Stripes/Vec.hpp>
