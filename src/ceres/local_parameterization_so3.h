#pragma once
#include <ceres/local_parameterization.h>

class LocalParameterizationSO3 : public ceres::LocalParameterization {
public:
    LocalParameterizationSO3();
    virtual ~LocalParameterizationSO3();

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

class LocalParameterizationSO3 : public ceres::LocalParameterization {
public:
    LocalParameterizationSO3();
    virtual ~LocalParameterizationSO3();

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

}
