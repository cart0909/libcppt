#pragma once
#include <Eigen/Dense>
#include <sophus/so3.hpp>

namespace Sophus {

Eigen::Matrix3d JacobianR(const Eigen::Vector3d& w);
Eigen::Matrix3d JacobianRInv(const Eigen::Vector3d& w);
Eigen::Matrix3d JacobianL(const Eigen::Vector3d& w);
Eigen::Matrix3d JacobianLInv(const Eigen::Vector3d& w);

};
