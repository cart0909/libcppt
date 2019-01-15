#pragma once
#include <vector>
#include <memory>
#include <chrono>
#include <Eigen/Dense>
#include <exception>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

#include <fcntl.h>
#include <iostream>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#define ENABLE_TRACE 1

// image pyramid
using ImagePyr = std::vector<cv::Mat>;

// eigen vector vector
#define STD_EIGEN_VECTOR(SIZE) \
using VecVector##SIZE##f = std::vector<Eigen::Vector##SIZE##f, Eigen::aligned_allocator<Eigen::Vector##SIZE##f>>; \
using VecVector##SIZE##d = std::vector<Eigen::Vector##SIZE##d, Eigen::aligned_allocator<Eigen::Vector##SIZE##d>>;

#define STD_EIGEN_DEQUE(SIZE) \
using DeqVector##SIZE##f = std::deque<Eigen::Vector##SIZE##f, Eigen::aligned_allocator<Eigen::Vector##SIZE##f>>; \
using DeqVector##SIZE##d = std::deque<Eigen::Vector##SIZE##d, Eigen::aligned_allocator<Eigen::Vector##SIZE##d>>;

// eigen extra matrix
#define EIGEN_EXTRA_MATRIX(SIZE1, SIZE2) \
using Matrix##SIZE1##_##SIZE2##d = Eigen::Matrix<double, SIZE1, SIZE2>; \
using Matrix##SIZE1##_##SIZE2##f = Eigen::Matrix<float, SIZE1, SIZE2>;

#define EIGEN_EXTRA_SQUARE_MATRIX(SIZE) \
using Matrix##SIZE##d = Eigen::Matrix<double, SIZE, SIZE>; \
using Matrix##SIZE##f = Eigen::Matrix<float, SIZE, SIZE>;

namespace Eigen {
EIGEN_EXTRA_MATRIX(2, 3)
EIGEN_EXTRA_MATRIX(3, 6)
STD_EIGEN_VECTOR(2)
STD_EIGEN_VECTOR(3)
STD_EIGEN_VECTOR(4)
STD_EIGEN_DEQUE(3)
EIGEN_EXTRA_SQUARE_MATRIX(6)
}

#define SMART_PTR(NAME) \
using NAME##Ptr = std::shared_ptr<NAME>; \
using NAME##ConstPtr = std::shared_ptr<const NAME>; \
using NAME##WPtr = std::weak_ptr<NAME>; \
using NAME##ConstWPtr = std::weak_ptr<const NAME>;

// simple timer
class TicToc {
public:
    inline TicToc() {
        tic();
    }

    inline void tic() {
        start = std::chrono::steady_clock::now();
    }

    inline double toc() {
        end = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed_seconds = end - start;
        return elapsed_seconds.count() * 1000; // ms
    }

private:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
};

// tracer
class Tracer {
public:
    inline Tracer()
    {
#if ENABLE_TRACE
        Init();
#endif
    }

    static inline void TraceCounter(const char* name, int32_t value)
    {
#if ENABLE_TRACE
        char buf[1024];
        snprintf(buf, 1024, "C|%d|%s|%d", getpid(), name, value);
        write(sTraceFD, buf, strlen(buf));
#endif
    }

    static inline void TraceBegin(const char* name)
    {
#if ENABLE_TRACE
        char buf[1024];
        size_t len = snprintf(buf, 1024, "B|%d|%s", getpid(), name);
        write(sTraceFD, buf, len);
#endif
    }

    static inline void TraceEnd()
    {
#if ENABLE_TRACE
        char buf = 'E';
        write(sTraceFD, &buf, 1);
#endif
    }

private:
    static inline void Init()
    {
#if ENABLE_TRACE
        const char* const traceFileName = "/sys/kernel/debug/tracing/trace_marker";
        sTraceFD = open(traceFileName, O_WRONLY);
        if (sTraceFD == -1) {
            std::cout << "error opening trace file: " << strerror(errno) << " (" << errno << ")" << std::endl;
            // sEnabledTags remains zero indicating that no tracing can occur
        }
#endif
    }

#if ENABLE_TRACE
    static int sTraceFD;
#endif
};

class ScopedTrace {
public:
    inline ScopedTrace(const char* name)
    {
#if ENABLE_TRACE
        Tracer::TraceBegin(name);
#endif
    }

    inline ~ScopedTrace()
    {
#if ENABLE_TRACE
        Tracer::TraceEnd();
#endif
    }
};

namespace Sophus {

Eigen::Matrix3d JacobianR(const Eigen::Vector3d& w);
Eigen::Matrix3d JacobianRInv(const Eigen::Vector3d& w);
Eigen::Matrix3d JacobianL(const Eigen::Vector3d& w);
Eigen::Matrix3d JacobianLInv(const Eigen::Vector3d& w);

};

namespace util {

// useful function
template<class T>
void ReduceVector(std::vector<T> &v, const std::vector<uchar>& status) {
    int j = 0;
    for(int i = 0, n = v.size(); i < n; ++i) {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

inline bool InBorder(const cv::Point2f& pt, int width, int height) {
    const int border_size = 1;
    int img_x = std::round(pt.x);
    int img_y = std::round(pt.y);
    if(img_x - border_size < 0 || img_y - border_size <= 0 || img_x + border_size >= width ||
            img_y + border_size >= height)
        return false;
    return true;
}

ImagePyr Pyramid(const cv::Mat& img, int num_levels);

}
