#include "local_parameterization_se3.h"
#include <sophus/se3.hpp>

LocalParameterizationSE3::LocalParameterizationSE3() {}
LocalParameterizationSE3::~LocalParameterizationSE3() {}

// Generalization of the addition operation,
//
//   x_plus_delta = Plus(x, delta)
//
// with the condition that Plus(x, 0) = x.
bool LocalParameterizationSE3::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Eigen::Vector3d> p(x + 4), dp(delta), omega(delta + 3);
    Eigen::Map<const Sophus::SO3d> q(x);
    Eigen::Map<Eigen::Vector3d> p_(x_plus_delta + 4);
    Eigen::Map<Sophus::SO3d> q_(x_plus_delta);
    Sophus::SO3d dq = Sophus::SO3d::exp(omega);
    p_ = p + dp;
    q_ = q * dq;
    return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
//
// jacobian is a row-major GlobalSize() x LocalSize() matrix.
bool LocalParameterizationSE3::ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    return true;
}

// Size of x.
int LocalParameterizationSE3::GlobalSize() const {
    return Sophus::SE3d::num_parameters;
}

// Size of delta.
int LocalParameterizationSE3::LocalSize() const {
    return Sophus::SE3d::DoF;
}
