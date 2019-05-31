#include "local_parameterization_so3.h"
#include <sophus/so3.hpp>

LocalParameterizationSO3::LocalParameterizationSO3() {}
LocalParameterizationSO3::~LocalParameterizationSO3() {}

// Generalization of the addition operation,
//
//   x_plus_delta = Plus(x, delta)
//
// with the condition that Plus(x, 0) = x.
bool LocalParameterizationSO3::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Sophus::SO3d> R(x);
    Eigen::Map<const Eigen::Vector3d> dx(delta);
    Eigen::Map<Sophus::SO3d> R_plus_dR(x_plus_delta);
    R_plus_dR = R * Sophus::SO3d::exp(dx);
    return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
//
// jacobian is a row-major GlobalSize() x LocalSize() matrix.
bool LocalParameterizationSO3::ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    return true;
}

// Size of x.
int LocalParameterizationSO3::GlobalSize() const {
    return Sophus::SO3d::num_parameters;
}

// Size of delta.
int LocalParameterizationSO3::LocalSize() const {
    return Sophus::SO3d::DoF;
}

namespace autodiff {

LocalParameterizationSO3::LocalParameterizationSO3() {}
LocalParameterizationSO3::~LocalParameterizationSO3() {}

// Generalization of the addition operation,
//
//   x_plus_delta = Plus(x, delta)
//
// with the condition that Plus(x, 0) = x.
bool LocalParameterizationSO3::Plus(const double* x, const double* delta, double* x_plus_delta) const {
    Eigen::Map<const Sophus::SO3d> R(x);
    Eigen::Map<const Eigen::Vector3d> dx(delta);
    Eigen::Map<Sophus::SO3d> R_plus_dR(x_plus_delta);
    R_plus_dR = R * Sophus::SO3d::exp(dx);
    return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
//
// jacobian is a row-major GlobalSize() x LocalSize() matrix.
bool LocalParameterizationSO3::ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<const Sophus::SO3d> R(x);
    Eigen::Map<Eigen::Matrix<double, 4, 3, Eigen::RowMajor>> J(jacobian);
    J = R.Dx_this_mul_exp_x_at_0();
    return true;
}

// Size of x.
int LocalParameterizationSO3::GlobalSize() const {
    return Sophus::SO3d::num_parameters;
}

// Size of delta.
int LocalParameterizationSO3::LocalSize() const {
    return Sophus::SO3d::DoF;
}

}
