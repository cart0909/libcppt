#pragma once
#include <Eigen/Dense>

class ImuData {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
    static Eigen::Matrix3d acc_noise_cov;
    static Eigen::Matrix3d gyr_noise_cov;
    static Eigen::Matrix3d acc_bias_cov; // random walk
    static Eigen::Matrix3d gyr_bias_cov; // random walk
    double timestamp;
};
