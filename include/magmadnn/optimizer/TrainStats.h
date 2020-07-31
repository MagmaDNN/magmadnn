#pragma once

#include "magmadnn.h"

namespace magmadnn {
namespace solver {

template <typename T>
struct TrainStats {
    int epoch;
    T loss;
    T accuracy;
    double time;
    int niter_step;

    TrainStats()
        : epoch(0),
          loss(static_cast<T>(0.0)),
          accuracy(static_cast<T>(0.0)),
          time(static_cast<double>(0.0)),
          niter_step(0) {}

    TrainStats(int epoch, T loss, T accuracy, double time, int niter_step = 0)
        : epoch(epoch), loss(loss), accuracy(accuracy), time(time), niter_step(niter_step) {}
};

}  // namespace solver
}  // namespace magmadnn
