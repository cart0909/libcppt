#include "local_parameterization_line.h"
#include "plucker/line.h"

LocalParameterizationLine3::LocalParameterizationLine3() {}
LocalParameterizationLine3::~LocalParameterizationLine3() {}

// Generalization of the addition operation,
//
//   x_plus_delta = Plus(x, delta)
//
// with the condition that Plus(x, 0) = x.
bool LocalParameterizationLine3::Plus(const double* x_raw,
                                      const double* delta_raw,
                                      double* x_plus_delta_raw) const {
    Eigen::Map<const Plucker::Line3d> x(x_raw);
    Eigen::Map<const Eigen::Vector4d> delta(delta_raw);
    Eigen::Map<Plucker::Line3d> x_plus_delta(x_plus_delta_raw);
    x_plus_delta = x * delta;
    return true;
}

// The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
//
// jacobian is a row-major GlobalSize() x LocalSize() matrix.
bool LocalParameterizationLine3::ComputeJacobian(const double* x, double* jacobian) const {
    Eigen::Map<Eigen::Matrix<double, 6, 4, Eigen::RowMajor>> J(jacobian);
    J.setIdentity();
    return true;
}

// Size of x.
int LocalParameterizationLine3::GlobalSize() const {
    return Plucker::Line3d::num_parameters;
}

// Size of delta.
int LocalParameterizationLine3::LocalSize() const {
    return Plucker::Line3d::DoF;
}
