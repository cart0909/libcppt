#pragma once
#include <Eigen/Dense>

class ImuData {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuData() {}
    ImuData(double acc_x, double acc_y, double acc_z,
            double gyr_x, double gyr_y, double gyr_z,
            double timestamp_)
        : acc(acc_x, acc_y, acc_z), gyr(gyr_x, gyr_y, gyr_z), timestamp(timestamp_) {}
    ImuData(const Eigen::Vector3d& acc_, const Eigen::Vector3d& gyr_, double timestamp_)
        : acc(acc_), gyr(gyr_), timestamp(timestamp_) {}

    Eigen::Vector3d acc;
    Eigen::Vector3d gyr;
    double timestamp;
};
