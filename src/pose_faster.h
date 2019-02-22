#pragma once
#include "util.h"
#include <mutex>
#include <atomic>

class PoseFaster {
public:
    PoseFaster(const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_, double g_norm);

    void UpdatePoseInfo(const Eigen::Vector3d& p_wb_, const Sophus::SO3d& q_wb_, const Eigen::Vector3d& v_wb_,
                        const Eigen::Vector3d& gyr_0_, const Eigen::Vector3d& acc_0_, double t0_,
                        const Eigen::Vector3d& ba_, const Eigen::Vector3d& bg_);

    bool Predict(const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double t,
                 Sophus::SE3d& Twc);

private:
    Eigen::Vector3d p_bc;
    Sophus::SO3d q_bc;

    std::mutex mtx_update;
    bool no_pose;
    bool update_pose;

    Eigen::Vector3d p_wb;
    Sophus::SO3d q_wb;
    Eigen::Vector3d v_wb;
    double pose_t;
    Eigen::Vector3d acc_0, gyr_0;
    Eigen::Vector3d ba, bg;

    Eigen::DeqVector3d d_acc, d_gyr;
    std::deque<double> d_imu_t;

    Eigen::Vector3d gw;
};
SMART_PTR(PoseFaster);
