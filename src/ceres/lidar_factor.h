#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/sized_cost_function.h>
#include <ceres/autodiff_cost_function.h>
#include "util.h"
class LidarEdgeFactor : public ceres::SizedCostFunction<3, 7, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarEdgeFactor(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                    const Eigen::Vector3d& last_point_b_, const Sophus::SE3d& Tlb_, double s_, double noise_);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    Sophus::SE3d Tlb;
    double s;
    Eigen::Matrix3d sqrt_info;
};

class LidarPlaneFactor : public ceres::SizedCostFunction<1, 7, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarPlaneFactor(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                    const Eigen::Vector3d& last_point_b_, const Eigen::Vector3d& last_point_d_,
                     const Sophus::SE3d& Tlb_, double s_, double noise_);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d curr_point, last_point_a, last_point_b, last_point_d, ljm_norm;
    Sophus::SE3d Tlb;
    double s, noise;
    Eigen::Matrix3d sqrt_info;
};


class LidarEdgeFactorIJ : public ceres::SizedCostFunction<3, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarEdgeFactorIJ(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                      const Eigen::Vector3d& last_point_b_, double s_);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d curr_point, last_point_a, last_point_b;
    Sophus::SE3d Tlb;
    double s;
    Eigen::Matrix3d sqrt_info;
};

class LidarPlaneFactorIJ : public ceres::SizedCostFunction<1, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarPlaneFactorIJ(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                       const Eigen::Vector3d& last_point_b_, const Eigen::Vector3d& last_point_d_, double s_);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d curr_point, last_point_a, last_point_b, last_point_d, ljm_norm;
    Sophus::SE3d Tlb;
    double s;
    Eigen::Matrix3d sqrt_info;
};
