#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>

class ImuState {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    uint64_t id;
    static uint64_t g_next_id;
    double timestamp;

    Sophus::SO3d qig;
    Eigen::Vector3d vgi;
    Eigen::Vector3d pgi;

    Eigen::Vector3d gyro_bias;
    Eigen::Vector3d acc_bias;

    Sophus::SO3d qic;
    Eigen::Vector3d pci; // ???

    static Eigen::Vector3d g_gravity;
};
