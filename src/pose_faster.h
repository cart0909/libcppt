#pragma once
#include "util.h"
#include <mutex>
#include <atomic>

class PoseFaster {
public:
    PoseFaster(const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_) {
        no_pose = true;
    }

    void UpdatePoseInfo(const Eigen::Vector3d& p_wb, const Sophus::SO3d& q_wb,
                        const Eigen::Vector3d& ba_, const Eigen::Vector3d& bg_,
                        double pose_t)
    {
        ba = ba_;
        bg = bg_;

        predict_p_wb = p_wb;
        predict_q_wb = q_wb;
        predict_pose_t = pose_t;

        int begin_index = -1;
        for(int i = 0, n = d_imu_t.size(); i < n; ++i) {
            if(predict_pose_t < d_imu_t[i]) {
                begin_index = i;
                break;
            }
        }
        begin_index -= 1;

        if(begin_index >= 0) {
            d_acc.erase(d_acc.begin(), d_acc.begin() + begin_index);
            d_gyr.erase(d_gyr.begin(), d_gyr.begin() + begin_index);
            d_imu_t.erase(d_imu_t.begin(), d_imu_t.begin() + begin_index);
        }

        if(no_pose)
            no_pose = false;
    }

    bool Predict(const Eigen::Vector3d& acc, const Eigen::Vector3d& gyr, double t,
                 Sophus::SE3d& Twc)
    {
        d_acc.emplace_back(acc);
        d_gyr.emplace_back(gyr);
        d_imu_t.emplace_back(t);

        if(no_pose) {
            return false;
        }
        else {
            return true;
        }
    }

private:
    void PredictNextPose(const Eigen::Vector3d& gyr_0, const Eigen::Vector3d& acc_0, double t0,
                         const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double t)
    {
//        Eigen::Vector3d gyr_0 = ref_frame->v_gyr.back(), acc_0 = ref_frame->v_acc.back();
//        double t0 = ref_frame->v_imu_timestamp.back();

//        for(int i = 0, n = cur_frame->v_acc.size(); i < n; ++i) {
//            double t = cur_frame->v_imu_timestamp[i], dt = t - t0;
//            Eigen::Vector3d gyr = cur_frame->v_gyr[i];
//            Eigen::Vector3d acc = cur_frame->v_acc[i];
//            cur_frame->imupreinte->push_back(dt, acc, gyr);

//            Eigen::Vector3d un_acc_0 = cur_frame->q_wb * (acc_0 - ref_frame->ba) - gw;
//            Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr) - ref_frame->bg;
//            cur_frame->q_wb = cur_frame->q_wb * Sophus::SO3d::exp(un_gyr * dt);
//            Eigen::Vector3d un_acc_1 = cur_frame->q_wb * (acc - ref_frame->ba) - gw;
//            Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
//            cur_frame->p_wb += dt * cur_frame->v_wb + 0.5 * dt * dt * un_acc;
//            cur_frame->v_wb += dt * un_acc;

//            gyr_0 = gyr;
//            acc_0 = acc;
//            t0 = t;
//        }
    }

    Eigen::Vector3d p_bc;
    Sophus::SO3d q_bc;

    std::atomic<bool> no_pose;
    std::mutex mtx_bias;
    Eigen::Vector3d ba;
    Eigen::Vector3d bg;

    std::mutex mtx_predict;
    Eigen::Vector3d predict_p_wb;
    Sophus::SO3d predict_q_wb;
    double predict_pose_t;

    std::mutex mtx_imu_buffer;
    Eigen::DeqVector3d d_acc, d_gyr;
    std::deque<double> d_imu_t;
};
SMART_PTR(PoseFaster);
