#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/sized_cost_function.h>
#include <ceres/autodiff_cost_function.h>

class LineProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length);

    bool Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const;

    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d spt, ept;
};

class LineSlaveProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 6>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSlaveProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length);

    bool Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const;

    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d spt, ept;
};

namespace autodiff {

class LineProjectionFactor {
public:
    LineProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length)
        : spt(spt_), ept(ept_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
    }

    template<class T>
    bool operator()(const T* Twb_raw, const T* Tbc_raw, const T* line_raw, T* residuals_raw) const {
        Eigen::Map<const Sophus::SE3<T>> Twb(Twb_raw), Tbc(Tbc_raw);
        Eigen::Map<const Sophus::SO3<T>> so3(line_raw);
        Eigen::Map<const Sophus::SO2<T>> so2(line_raw + 4);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Sophus::SE3<T> Tcw = (Twb * Tbc).inverse();
        Eigen::Matrix<T, 3, 3> U = so3.matrix();
        Eigen::Matrix<T, 2, 1> W = so2.unit_complex();
        Eigen::Matrix<T, 3, 1> mw = U.col(0) * W(0), lw = U.col(1) * W(1);
        Eigen::Matrix<T, 3, 1> mc = Tcw.so3() * mw + Sophus::SO3<T>::hat(Tcw.translation()) * (Tcw.so3() * lw);
        Eigen::Matrix<T, 3, 1> l = mc / mc.template head<2>().norm();

        residuals << l.dot(spt.cast<T>()),
                     l.dot(ept.cast<T>());

        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& spt, const Eigen::Vector3d& ept, double focal_length) {
        return new ceres::AutoDiffCostFunction<LineProjectionFactor, 2, 7, 7, 6>(
                    new LineProjectionFactor(spt, ept, focal_length));
    }
    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d spt, ept;
};

class LineSlaveProjectionFactor {
public:
    LineSlaveProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length)
        : spt(spt_), ept(ept_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
    }

    template<class T>
    bool operator()(const T* Twb_raw, const T* Tbc_raw, const T* Tsm_raw, const T* line_raw, T* residuals_raw) const {
        Eigen::Map<const Sophus::SE3<T>> Twb(Twb_raw), Tbc(Tbc_raw), Tsm(Tsm_raw);
        Eigen::Map<const Sophus::SO3<T>> so3(line_raw);
        Eigen::Map<const Sophus::SO2<T>> so2(line_raw + 4);
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Sophus::SE3<T> Tsw = Tsm * (Twb * Tbc).inverse();
        Eigen::Matrix<T, 3, 3> U = so3.matrix();
        Eigen::Matrix<T, 2, 1> W = so2.unit_complex();
        Eigen::Matrix<T, 3, 1> mw = U.col(0) * W(0), lw = U.col(1) * W(1);
        Eigen::Matrix<T, 3, 1> ms = Tsw.so3() * mw + Sophus::SO3<T>::hat(Tsw.translation()) * (Tsw.so3() * lw);
        Eigen::Matrix<T, 3, 1> l = ms / ms.template head<2>().norm();

        residuals << l.dot(spt.cast<T>()),
                     l.dot(ept.cast<T>());

        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& spt, const Eigen::Vector3d& ept, double focal_length) {
        return new ceres::AutoDiffCostFunction<LineSlaveProjectionFactor, 2, 7, 7, 7, 6>(
                    new LineSlaveProjectionFactor(spt, ept, focal_length));
    }
    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d spt, ept;
};
}
