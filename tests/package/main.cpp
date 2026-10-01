//----------------------------------------------------------------------------------------------
// Copyright (c) The Einsums Developers. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
//----------------------------------------------------------------------------------------------

#include <Stripes/RungLadder.hpp>
#include <Stripes/RuntimeFeatures.hpp>
#include <cmath>
#include <cstddef>
#include <cstdio>

#define DECLARE_RUNG(ns)                                                                                                                   \
    namespace ns {                                                                                                                         \
    int  double_lanes();                                                                                                                   \
    void exp_all(double const *x, double *y, std::size_t n);                                                                               \
    }
STRIPES_FOR_EACH_BUILT_RUNG(DECLARE_RUNG)

int main() {
    using LanesFn       = int (*)();
    using ExpFn         = void (*)(double const *, double *, std::size_t);
    LanesFn const lanes = stripes::select<LanesFn>(STRIPES_LADDER(double_lanes));
    ExpFn const   exp   = stripes::select<ExpFn>(STRIPES_LADDER(exp_all));

    double x[11], y[11];
    for (int i = 0; i < 11; ++i) {
        x[i] = 0.25 * i - 1.0;
    }
    exp(x, y, 11);
    int failures = 0;
    for (int i = 0; i < 11; ++i) {
        failures += std::fabs(y[i] - std::exp(x[i])) > 1e-15 * std::exp(x[i]);
    }
    std::printf("rung %s, %d double lanes, exp %s\n", stripes::to_string(stripes::selected_arch()), lanes(), failures ? "FAILED" : "ok");
    return failures ? 1 : 0;
}
