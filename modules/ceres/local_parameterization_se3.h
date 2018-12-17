#pragma once
#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <ros/ros.h>

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
//        Eigen::Map<const SE3d> T(T_raw);
//        Eigen::Map<const Vector6d> delta(delta_raw);
//        Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
//        T_plus_delta = T * SE3d::exp(delta);

        Eigen::Map<const SE3d> T(T_raw);
        Eigen::Map<const Vector3d> dx(delta_raw);
        Eigen::Map<const Vector3d> dq(delta_raw + 3);
        Eigen::Map<SE3d> T_plus_delta(T_plus_delta_raw);
        T_plus_delta.so3() = T.so3() * SO3d::exp(dq);
        T_plus_delta.translation() = T.translation() + dx;
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
        Eigen::Map<const Sophus::SE3d> T(T_raw);
        Eigen::Map<Eigen::Matrix<double, 7, 6, Eigen::RowMajor> > jacobian(jacobian_raw);
        jacobian = T.Dx_this_mul_exp_x_at_0();
        return true;
    }

    virtual int GlobalSize() const {
        return SE3d::num_parameters;
    }

    virtual int LocalSize() const {
        return SE3d::DoF;
    }

    static ceres::LocalParameterization* Create() {
        return new AutoDiff::VertexSE3();
    }
};

}
}
