#pragma once
#include "util.h"
#include "imu_preintegration.h"
#include <ceres/sized_cost_function.h>

class ImuFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9> {
public:
    ImuFactor(ImuPreintegrationPtr preintegration_, const Eigen::Vector3d& gw_,
              const Eigen::Matrix6d& inv_cov_acc_gyr_bias);

    bool Evaluate(double const *const *parameters_raw, double *residuals_raw, double **jacobians_raw) const;

    ImuPreintegrationPtr preintegration;
    Eigen::Vector3d gw;
    Eigen::Matrix15d sqrt_info;
};
