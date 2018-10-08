#pragma once
#include <Eigen/Dense>
#include <sophus/so3.hpp>

class CamState {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    double timestamp;
    Sophus::SO3d qcg;
    Eigen::Vector3d pgc;
};
