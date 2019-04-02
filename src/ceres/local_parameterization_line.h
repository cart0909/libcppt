#pragma once
#include <ceres/local_parameterization.h>
#include <ceres/autodiff_local_parameterization.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>

class LocalParameterizationLine3 : public ceres::LocalParameterization {
public:
    LocalParameterizationLine3();
    virtual ~LocalParameterizationLine3();

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

class LocalParameterizationSO3xSO2 : public ceres::LocalParameterization {
public:
    LocalParameterizationSO3xSO2();
    virtual ~LocalParameterizationSO3xSO2();

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

class LocalParameterizationSO3xSO2 {
public:
    template<class T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
        Eigen::Map<const Sophus::SO3<T>> U(x);
        Eigen::Map<const Sophus::SO2<T>> W(x + 4);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> dU(delta);
        T dW = delta[3];
        Eigen::Map<Sophus::SO3<T>> U_plus_dU(x_plus_delta);
        Eigen::Map<Sophus::SO2<T>> W_plus_dW(x_plus_delta + 4);

        U_plus_dU = U * Sophus::SO3<T>::exp(dU);
        W_plus_dW = W * Sophus::SO2<T>::exp(dW);
        return true;
    }

    static ceres::LocalParameterization* Create() {
        return new ceres::AutoDiffLocalParameterization<LocalParameterizationSO3xSO2, 6, 4>;
    }
};

}
