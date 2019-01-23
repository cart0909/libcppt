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

class IntegrationBase
{
  public:
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

//    Eigen::Matrix<double, 15, 1> evaluate(const Eigen::Vector3d &Pi, const Eigen::Quaterniond &Qi, const Eigen::Vector3d &Vi, const Eigen::Vector3d &Bai, const Eigen::Vector3d &Bgi,
//                                          const Eigen::Vector3d &Pj, const Eigen::Quaterniond &Qj, const Eigen::Vector3d &Vj, const Eigen::Vector3d &Baj, const Eigen::Vector3d &Bgj)
//    {
//        Eigen::Matrix<double, 15, 1> residuals;

//        Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
//        Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

//        Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

//        Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
//        Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

//        Eigen::Vector3d dba = Bai - linearized_ba;
//        Eigen::Vector3d dbg = Bgi - linearized_bg;

//        Eigen::Quaterniond corrected_delta_q = delta_q * Utility::deltaQ(dq_dbg * dbg);
//        Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
//        Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

//        residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * G * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
//        residuals.block<3, 1>(O_R, 0) = 2 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).vec();
//        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (G * sum_dt + Vj - Vi) - corrected_delta_v;
//        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
//        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
//        return residuals;
//    }

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
    std::vector<Eigen::Vector3d> acc_buf;
    std::vector<Eigen::Vector3d> gyr_buf;
};

