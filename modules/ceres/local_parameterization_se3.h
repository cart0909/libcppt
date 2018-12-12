#pragma once
#include <ceres/ceres.h>
#include <sophus/se3.hpp>

namespace Sophus {
class VertexSE3 : public ceres::LocalParameterization {
public:
    VertexSE3() {}
    ~VertexSE3() {}

    // SE3 plus opration for Ceres
    //
    // T * exp(x)
    //
    virtual bool Plus(double const* T_raw, double const* delta_raw,
                      double* T_plus_delta_raw) const {
        Eigen::Map<const SE3d> T(T_raw);
        Eigen::Map<const Vector6d> delta(delta_raw);
        Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = T * SE3d::exp(delta);
        return true;
    }

    // Jacobian of SE3 plus operation for Ceres
    //
    // Dx T * exp(x) with x = 0
    //
    virtual bool ComputeJacobian(double const* T_raw, double* jacobian_raw) const {
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > jacobian(jacobian_raw);
        jacobian.setIdentity();
        return true;
    }

    virtual int GlobalSize() const {
        return SE3d::num_parameters;
    }

    virtual int LocalSize() const {
        return SE3d::DoF;
    }

    static ceres::LocalParameterization* Create() {
        return new VertexSE3();
    }
};

namespace AutoDiff {

class VertexSE3 {
public:
    VertexSE3() {}
    ~VertexSE3() {}

    template<class T>
    bool operator() (const T* T_raw, const T* delta_raw, T* T_plus_delta_raw) const {
        using Vector6T = Eigen::Matrix<T, 6, 1>;
        Eigen::Map<const SE3<T>> mT(T_raw);
        Eigen::Map<const Vector6T> delta(delta_raw);
        Eigen::Map<SE3<T>> T_plus_delta(T_plus_delta_raw);
        T_plus_delta = mT * SE3<T>::exp(delta);
        return true;
    }

    static ceres::LocalParameterization* Create() {
        return new ceres::AutoDiffLocalParameterization<VertexSE3, 7, 6>;
    }
};

}
}
