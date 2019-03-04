#pragma once
#include <ceres/sized_cost_function.h>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "util.h"

class ProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionTdFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& velocity_i_, double td_i_,
                       const Eigen::Vector3d& pt_j_, const Eigen::Vector3d& velocity_j_, double td_j_,
                       double focal_length);

    bool Evaluate(double const * const *parameters_raw, double* residuals_raw, double** jacobians_raw) const;
    double td_i, td_j;
    Eigen::Vector3d pt_i, pt_j;
    Eigen::Vector3d velocity_i, velocity_j;
    Eigen::Matrix2d sqrt_info;
};

class SlaveProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SlaveProjectionTdFactor(const Eigen::Vector3d& pt_mi_, const Eigen::Vector3d& velocity_i_, double td_i_,
                            const Eigen::Vector3d& pt_sj_, const Eigen::Vector3d& velocity_j_, double td_j_,
                            double focal_length);

    bool Evaluate(double const * const *parameters_raw, double* residuals_raw, double** jacobians_raw) const;
    double td_i, td_j;
    Eigen::Vector3d pt_mi, pt_sj;
    Eigen::Vector3d velocity_i, velocity_j;
    Eigen::Matrix2d sqrt_info;
};

class SelfProjectionTdFactor : public ceres::SizedCostFunction<2, 7, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SelfProjectionTdFactor(const Eigen::Vector3d& pt_l_, const Eigen::Vector3d& velocity_l_,
                           const Eigen::Vector3d& pt_r_, const Eigen::Vector3d& velocity_r_,
                           double td_0_, double focal_length);

    bool Evaluate(double const * const *parameters_raw, double* residuals_raw, double** jacobians_raw) const;

    double td_0;
    Eigen::Vector3d pt_l, pt_r;
    Eigen::Vector3d velocity_l, velocity_r;
    Eigen::Matrix2d sqrt_info;
};
