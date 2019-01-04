#pragma once
#include "util.h"

class Camera {
public:
    template<class Scalar>
    void Project(const Eigen::Matrix<Scalar, 3, 1>& P,
                 Eigen::Matrix<Scalar, 2, 1>& p) const;

    const int width;
    const int height;
    const double fx;
    const double fy;
    const double cx;
    const double cy;

    // rectify table
    cv::Mat M1, M2;
};
