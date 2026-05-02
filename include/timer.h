#pragma once
#include <chrono>

class Timer {
    using Clock = std::chrono::high_resolution_clock;
    std::chrono::time_point<Clock> t_start;
public:
    void   start()       { t_start = Clock::now(); }
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(
            Clock::now() - t_start).count();
    }
};
