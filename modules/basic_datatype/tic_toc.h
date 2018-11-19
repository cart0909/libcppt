#pragma once
#include <chrono>

class TicToc {
public:
    TicToc() {
        tic();
    }

    void tic() {
        start = std::chrono::steady_clock::now();
    }

    double toc() {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000; // ms
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
};
