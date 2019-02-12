/*******************************************************
 * Copyright (C) 2019, Aerial Robotics Group, Hong Kong University of Science and Technology
 * 
 * This file is part of VINS.
 * 
 * Licensed under the GNU General Public License v3.0;
 * you may not use this file except in compliance with the License.
 *******************************************************/
#pragma once
#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <ceres/ceres.h>
#include "util.h"

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12
};

enum NoiseOrder
{
    O_AN = 0,
    O_GN = 3,
    O_AW = 6,
    O_GW = 9
};


class IntegrationBase
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    IntegrationBase() = delete;
    IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                    const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg,
                    double acc_n, double gyr_n, double acc_w, double gyr_w);

    void push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr);
    void repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg);

    void midPointIntegration(double _dt, 
                            const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0,
                            const Eigen::Vector3d& _acc_1, const Eigen::Vector3d& _gyr_1,
                            const Eigen::Vector3d& delta_p, const Sophus::SO3d& delta_q, const Eigen::Vector3d& delta_v,
                            const Eigen::Vector3d& linearized_ba, const Eigen::Vector3d& linearized_bg,
                            Eigen::Vector3d& result_delta_p, Sophus::SO3d& result_delta_q, Eigen::Vector3d& result_delta_v,
                            Eigen::Vector3d& result_linearized_ba, Eigen::Vector3d& result_linearized_bg, bool update_jacobian);

    void propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1);

    Eigen::Vector15d evaluate(const Eigen::Vector3d& Pi, const Sophus::SO3d& Qi, const Eigen::Vector3d& Vi, const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi,
                              const Eigen::Vector3d& Pj, const Sophus::SO3d& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj, const Eigen::Vector3d& Bgj,
                              const Eigen::Vector3d& Gw);

    double dt;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d acc_1, gyr_1;

    const Eigen::Vector3d linearized_acc, linearized_gyr;
    Eigen::Vector3d linearized_ba, linearized_bg;

    Eigen::Matrix15d jacobian, covariance;
    Eigen::Matrix18d noise;

    double sum_dt;
    Eigen::Vector3d delta_p;
    Sophus::SO3d delta_q;
    Eigen::Vector3d delta_v;

    std::vector<double> dt_buf;
    Eigen::VecVector3d acc_buf;
    Eigen::VecVector3d gyr_buf;
};
SMART_PTR(IntegrationBase)
