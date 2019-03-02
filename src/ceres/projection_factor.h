#pragma once
#include <ceres/sized_cost_function.h>
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
