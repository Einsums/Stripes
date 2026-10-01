//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

// Compiled once per rung, each copy in its own STRIPES_ARCH_NS.

#include <Stripes/Math.hpp>
#include <Stripes/Partial.hpp>
#include <Stripes/Vec.hpp>
#include <cstddef>

namespace STRIPES_ARCH_NS {

int double_lanes() {
    return stripes::lanes<double>;
}

void exp_all(double const *x, double *y, std::size_t n) {
    constexpr std::size_t L = stripes::lanes<double>;
    std::size_t           i = 0;
    for (; i + L <= n; i += L) {
        stripes::storeu(y + i, stripes::exp(stripes::loadu(x + i)));
    }
    if (i < n) {
        stripes::storeu_partial(y + i, stripes::exp(stripes::loadu_partial(x + i, n - i)), n - i);
    }
}

} // namespace STRIPES_ARCH_NS
