#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "basic_datatype/util_datatype.h"
#include "basic_datatype/basic_sensor.h"

class SimpleStereoCam : public SensorBase {
public:
    SimpleStereoCam() = delete;
    SimpleStereoCam(const Sophus::SE3d& Tbs_, int width_, int height_,
                    double f_, double cx_, double cy_, double b_,
                    const cv::Mat& M1l_, const cv::Mat& M2l_,
                    const cv::Mat& M1r_, const cv::Mat& M2r_);
    virtual ~SimpleStereoCam();

    // left cam api
    template<class Scalar>
    void Project2(const Eigen::Matrix<Scalar, 3, 1>& P,
                  Eigen::Matrix<Scalar, 2, 1>& p,
                  Eigen::Matrix<Scalar, 2, 3>* J = nullptr) const;

    template<class Scalar>
    Eigen::Matrix<Scalar, 2, 3> J2(const Eigen::Matrix<Scalar, 3, 1>& P) const;

    // P(2) = 1
    template<class Scalar>
    void BackProject(const Eigen::Matrix<Scalar, 2, 1>& p,
                     Eigen::Matrix<Scalar, 3, 1>& P) const;

    // stereo cam api
    // p is [u, v, ur]'
    template<class Scalar>
    void Project3(const Eigen::Matrix<Scalar, 3, 1>& P,
                  Eigen::Matrix<Scalar, 3, 1>& p,
                  Eigen::Matrix<Scalar, 3, 3>* J = nullptr) const;

    template<class Scalar>
    Eigen::Matrix<Scalar, 3, 3> J3(const Eigen::Matrix<Scalar, 3, 1>& P) const;

    // p is [u, v, ur]'
    template<class Scalar>
    void Triangulate(const Eigen::Matrix<Scalar, 3, 1>& p,
                     Eigen::Matrix<Scalar, 3, 1>& P) const;


    const int width;
    const int height;
    const double f;
    const double inv_f;
    const double cx;
    const double cy;
    const double b;
    const double bf;

    // rectify table
    cv::Mat M1l, M2l;
    cv::Mat M1r, M2r;
};

SMART_PTR(SimpleStereoCam)

// left cam api
template<class Scalar>
void SimpleStereoCam::Project2(const Eigen::Matrix<Scalar, 3, 1>& P,
                               Eigen::Matrix<Scalar, 2, 1>& p,
                               Eigen::Matrix<Scalar, 2, 3>* J) const {
    Scalar inv_z = Scalar(1.0) / P(2);
    Scalar f_inv_z = Scalar(f) * inv_z;
    p(0) = P(0) * f_inv_z + Scalar(cx);
    p(1) = P(1) * f_inv_z + Scalar(cy);

    if(J) {
        Scalar f_inv_z2 = f_inv_z * inv_z;
        *J << f_inv_z, 0, -f_inv_z2 * P(0),
              0, f_inv_z, -f_inv_z2 * P(1);
    }
}

template<class Scalar>
Eigen::Matrix<Scalar, 2, 3> SimpleStereoCam::J2(const Eigen::Matrix<Scalar, 3, 1>& P) const {
    Eigen::Matrix<Scalar, 2, 3> J;
    Scalar inv_z = Scalar(1.0) / P(2);
    Scalar f_inv_z = f * inv_z;
    Scalar f_inv_z2 = f_inv_z * inv_z;

    J << f_inv_z, 0, -f_inv_z2 * P(0),
         0, f_inv_z, -f_inv_z2 * P(1);
    return J;
}

// P(2) = 1
template<class Scalar>
void SimpleStereoCam::BackProject(const Eigen::Matrix<Scalar, 2, 1>& p,
                                  Eigen::Matrix<Scalar, 3, 1>& P) const {
    P << (p(0) - Scalar(cx)) * Scalar(inv_f), (p(1) - Scalar(cy)) * Scalar(inv_f), 1;
}

// stereo cam api
// p is [u, v, ur]'
template<class Scalar>
void SimpleStereoCam::Project3(const Eigen::Matrix<Scalar, 3, 1>& P,
                               Eigen::Matrix<Scalar, 3, 1>& p,
                               Eigen::Matrix<Scalar, 3, 3>* J) const {
    Scalar inv_z = Scalar(1.0) / P(2);
    Scalar f_inv_z = Scalar(f) * inv_z;
    p(0) = P(0) * f_inv_z + Scalar(cx); // u
    p(1) = P(1) * f_inv_z + Scalar(cy); // v
    p(2) = p(0) - Scalar(bf) * inv_z;   // ur

    if(J) {
        Scalar inv_z2 = inv_z * inv_z;
        Scalar f_inv_z2 = Scalar(f) * inv_z2;
        *J << f_inv_z,       0,         -f_inv_z2 * P(0),
                    0, f_inv_z,         -f_inv_z2 * P(1),
              f_inv_z,       0, (Scalar(bf) - Scalar(f) * P(0)) * inv_z2;
    }
}

template<class Scalar>
Eigen::Matrix<Scalar, 3, 3> SimpleStereoCam::J3(const Eigen::Matrix<Scalar, 3, 1>& P) const {
    Eigen::Matrix<Scalar, 3, 3> J;
    Scalar inv_z = Scalar(1.0) / P(2);
    Scalar inv_z2 = inv_z * inv_z;
    Scalar f_inv_z = Scalar(f) * inv_z;
    Scalar f_inv_z2 = Scalar(f) * inv_z2;

    J << f_inv_z,       0,         -f_inv_z2 * P(0),
               0, f_inv_z,          -f_inv_z2 * P(1),
         f_inv_z,       0, (Scalar(bf) - Scalar(f) * P(0)) * inv_z2;
    return J;
}

// p is [u, v, ur]'
template<class Scalar>
void SimpleStereoCam::Triangulate(const Eigen::Matrix<Scalar, 3, 1>& p,
                                  Eigen::Matrix<Scalar, 3, 1>& P) const {
    Scalar disparity = p(0) - p(2); // ul - ur
    Scalar z = Scalar(bf) / disparity;
    P << (p(0) - Scalar(cx)) * Scalar(inv_f), (p(1) - Scalar(cy)) * Scalar(inv_f), 1;
    P *= z;
}
