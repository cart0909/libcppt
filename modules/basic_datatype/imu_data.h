#pragma once
#include <Eigen/Core>

class ImuData {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    double timestamp;
};
