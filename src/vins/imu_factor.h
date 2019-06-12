/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/

#pragma once
#include <ros/assert.h>
#include <iostream>
#include <Eigen/Dense>
#include "integration_base.h"

#include <ceres/ceres.h>

class IMUFactor : public ceres::SizedCostFunction<15, 7, 9, 7, 9>
{
  public:
    IMUFactor() = delete;
    IMUFactor(IntegrationBasePtr _pre_integration, const Eigen::Vector3d& Gw_);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    IntegrationBasePtr pre_integration;
    Eigen::Vector3d Gw;
};


//only consider posej
class IMUFactorForLidar : public ceres::SizedCostFunction<15, 7, 9>
{
  public:
    IMUFactorForLidar() = delete;
    IMUFactorForLidar(IntegrationBasePtr _pre_integration, const Eigen::Vector3d& Gw_);
    virtual bool Evaluate(double const *const *parameters, double *residuals, double **jacobians) const;

    IntegrationBasePtr pre_integration;
    Eigen::Vector3d Gw;
};
