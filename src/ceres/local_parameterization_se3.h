#pragma once
#include <ceres/local_parameterization.h>

class LocalParameterizationSE3 : public ceres::LocalParameterization {
public:
    LocalParameterizationSE3();
    virtual ~LocalParameterizationSE3();

    // Generalization of the addition operation,
    //
    //   x_plus_delta = Plus(x, delta)
    //
    // with the condition that Plus(x, 0) = x.
    virtual bool Plus(const double* x,
                      const double* delta,
                      double* x_plus_delta) const;

    // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
    //
    // jacobian is a row-major GlobalSize() x LocalSize() matrix.
    virtual bool ComputeJacobian(const double* x, double* jacobian) const;

    // Size of x.
    virtual int GlobalSize() const;

    // Size of delta.
    virtual int LocalSize() const;
};

namespace autodiff {

class LocalParameterizationSE3 : public ceres::LocalParameterization {
public:
    LocalParameterizationSE3();
    virtual ~LocalParameterizationSE3();

    // Generalization of the addition operation,
    //
    //   x_plus_delta = Plus(x, delta)
    //
    // with the condition that Plus(x, 0) = x.
    virtual bool Plus(const double* T_raw,
                      const double* delta_raw,
                      double* T_plus_delta_raw) const;

    // The jacobian of Plus(x, delta) w.r.t delta at delta = 0.
    //
    // jacobian is a row-major GlobalSize() x LocalSize() matrix.
    virtual bool ComputeJacobian(double const* T_raw,
                                 double* jacobian_raw) const;

    // Size of x.
    virtual int GlobalSize() const;

    // Size of delta.
    virtual int LocalSize() const;
};

};
