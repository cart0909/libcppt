#pragma once
#include <Eigen/Dense>
#include <ceres/sized_cost_function.h>

class LineProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineProjectionFactor(const Eigen::Vector3d& spi_, const Eigen::Vector3d& epi_,
                         const Eigen::Vector3d& spj_, const Eigen::Vector3d& epj_,
                         double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d spi, epi;
    Eigen::Vector3d spj, epj;
    Eigen::Matrix2d sqrt_info;
};

class LineSlaveProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSlaveProjectionFactor(const Eigen::Vector3d& sp_mi_, const Eigen::Vector3d& ep_mi_,
                              const Eigen::Vector3d& sp_sj_, const Eigen::Vector3d& ep_sj_,
                              double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d sp_mi, ep_mi;
    Eigen::Vector3d sp_sj, ep_sj;
    Eigen::Matrix2d sqrt_info;
};

class LineSelfProjectionFactor : public ceres::SizedCostFunction<2, 7, 1, 1>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSelfProjectionFactor(const Eigen::Vector3d& sp_l_, const Eigen::Vector3d& ep_l_,
                             const Eigen::Vector3d& sp_r_, const Eigen::Vector3d& ep_r_,
                             double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d spl, epl;
    Eigen::Vector3d spr, epr;
    Eigen::Matrix2d sqrt_info;
};
