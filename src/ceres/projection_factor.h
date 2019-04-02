#pragma once
#include <ceres/sized_cost_function.h>
#include <ceres/autodiff_cost_function.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "util.h"

// imaster -> jmaster
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& pt_j_,
                     const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_,
                     double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d pt_i, pt_j;
    Sophus::SO3d q_bc;
    Eigen::Vector3d p_bc;
    Eigen::Matrix2d sqrt_info;
};

// imaster -> jslave
class SlaveProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SlaveProjectionFactor(const Eigen::Vector3d& pt_mi_, const Eigen::Vector3d& pt_sj_,
                          const Sophus::SO3d& q_sm_, const Eigen::Vector3d& p_sm_,
                          const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_,
                          double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d pt_mi, pt_sj;
    Sophus::SO3d q_sm, q_bc;
    Eigen::Vector3d p_sm, p_bc;
    Eigen::Matrix2d sqrt_info;
};

// imaster -> islave
class SelfProjectionFactor : public ceres::SizedCostFunction<2, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SelfProjectionFactor(const Eigen::Vector3d& pt_l_, const Eigen::Vector3d& pt_r_,
                         const Sophus::SO3d& q_rl_, const Eigen::Vector3d& p_rl_,
                         double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d pt_l, pt_r;
    Sophus::SO3d q_rl;
    Eigen::Vector3d p_rl;
    Eigen::Matrix2d sqrt_info;
};

class ProjectionExFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionExFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& pt_j_,
                       double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d pt_i, pt_j;
    Eigen::Matrix2d sqrt_info;
};

class SlaveProjectionExFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SlaveProjectionExFactor(const Eigen::Vector3d& pt_mi_, const Eigen::Vector3d& pt_sj_,
                            double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d pt_mi, pt_sj;
    Eigen::Matrix2d sqrt_info;
};

class SelfProjectionExFactor : public ceres::SizedCostFunction<2, 7, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SelfProjectionExFactor(const Eigen::Vector3d& pt_l_, const Eigen::Vector3d& pt_r_,
                           double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d pt_l, pt_r;
    Eigen::Matrix2d sqrt_info;
};

namespace autodiff {

class ProjectionExFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionExFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& pt_j_, double focal_length)
        : pt_i(pt_i_), pt_j(pt_j_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
    }

    template<class T>
    bool operator()(const T* Twbi_raw, const T* Twbj_raw, const T* Tbc_raw, const T* inv_zi_raw,
                    T* residuals_raw) const
    {
        Eigen::Map<const Sophus::SE3<T>> Twbi(Twbi_raw), Twbj(Twbj_raw), Tbc(Tbc_raw);
        T inv_zi = inv_zi_raw[0];
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Eigen::Matrix<T, 3, 1> x3Dci = pt_i.cast<T>() / inv_zi;
        Eigen::Matrix<T, 3, 1> x3Dcj = Tbc.inverse() * Twbj.inverse() * Twbi * Tbc * x3Dci;
        T inv_zj = T(1.0f) / x3Dcj(2);
        residuals = x3Dcj.template head<2>() * inv_zj - pt_j.head<2>().cast<T>();
        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pt_i, const Eigen::Vector3d& pt_j, double focal_length) {
        return new ceres::AutoDiffCostFunction<autodiff::ProjectionExFactor, 2, 7, 7, 7, 1>(
                    new autodiff::ProjectionExFactor(pt_i, pt_j, focal_length));
    }

    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d pt_i, pt_j;
};

class SlaveProjectionExFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SlaveProjectionExFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& pt_j_, double focal_length)
        : pt_i(pt_i_), pt_j(pt_j_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
    }

    template<class T>
    bool operator()(const T* Twbi_raw, const T* Twbj_raw, const T* Tbc_raw, const T* Tsm_raw, const T* inv_zi_raw,
                    T* residuals_raw) const
    {
        Eigen::Map<const Sophus::SE3<T>> Twbi(Twbi_raw), Twbj(Twbj_raw), Tbc(Tbc_raw), Tsm(Tsm_raw);
        T inv_zi = inv_zi_raw[0];
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Eigen::Matrix<T, 3, 1> x3Dci = pt_i.cast<T>() / inv_zi;
        Eigen::Matrix<T, 3, 1> x3Dcj = Tsm * Tbc.inverse() * Twbj.inverse() * Twbi * Tbc * x3Dci;
        T inv_zj = T(1.0f) / x3Dcj(2);
        residuals = x3Dcj.template head<2>() * inv_zj - pt_j.head<2>().cast<T>();
        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pt_i, const Eigen::Vector3d& pt_j, double focal_length) {
        return new ceres::AutoDiffCostFunction<autodiff::SlaveProjectionExFactor, 2, 7, 7, 7, 7, 1>(
                    new autodiff::SlaveProjectionExFactor(pt_i, pt_j, focal_length));
    }

    Eigen::Matrix2d sqrt_info;
    Eigen::Vector3d pt_i, pt_j;
};

class SelfProjectionExFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SelfProjectionExFactor(const Eigen::Vector3d& pt_l_, const Eigen::Vector3d& pt_r_, double focal_length)
        : pt_l(pt_l_), pt_r(pt_r_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
    }

    template<class T>
    bool operator()(const T* Trl_raw, const T* inv_zl_raw, T* residuals_raw) const
    {
        Eigen::Map<const Sophus::SE3<T>> Trl(Trl_raw);
        T inv_zl = inv_zl_raw[0];
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Eigen::Matrix<T, 3, 1> x3Dl = pt_l.cast<T>() / inv_zl;
        Eigen::Matrix<T, 3, 1> x3Dr = Trl * x3Dl;
        T inv_zr = T(1.0f) / x3Dr(2);
        residuals = x3Dr.template head<2>() * inv_zr - pt_r.head<2>().cast<T>();
        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& pt_l, const Eigen::Vector3d& pt_r, double focal_length) {
        return new ceres::AutoDiffCostFunction<autodiff::SelfProjectionExFactor, 2, 7, 1>(
                    new autodiff::SelfProjectionExFactor(pt_l, pt_r, focal_length));
    }
    Eigen::Vector3d pt_l, pt_r;
    Eigen::Matrix2d sqrt_info;
};
}
