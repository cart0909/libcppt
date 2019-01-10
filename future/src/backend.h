#pragma once
#include <thread>
#include <condition_variable>
#include "util.h"
#include "imu_preintegration.h"

class BackEnd {
public:
    enum MarginType {
        MARGIN_OUT_NEW,
        MARGIN_OUT_OLD
    };

    struct Frame {
        uint64_t id;
        double timestamp;
        std::vector<uint64_t>  pt_id;
        Eigen::VecVector2d pt, pt_r;
        Eigen::VecVector3d pt_normal_plane, pt_r_normal_plane;

        Sophus::SO3d    q_wb;
        Eigen::Vector3d p_wb;
        Eigen::Vector3d v_wb;

        Eigen::Vector3d ba;
        Eigen::Vector3d bg;

        Eigen::VecVector3d v_gyr, v_acc;
        ImuPreintegrationPtr imu_preintegration;
    };
    SMART_PTR(Frame)

    BackEnd(double focal_length_,
            double gyr_n, double acc_n,
            double gyr_w, double acc_w,
            const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
            const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_);
    ~BackEnd();

    void PushFrame(FramePtr frame);
    void Process();

private:
    std::thread thread_;
    std::mutex  m_buffer;
    std::condition_variable cv_buffer;
    std::deque<FramePtr> frame_buffer;

    double focal_length;
    Eigen::Vector3d p_rl, p_bc;
    Sophus::SO3d    q_rl, q_bc;

    Eigen::Matrix3d gyr_noise_cov;
    Eigen::Matrix3d acc_noise_cov;
    Eigen::Matrix3d gyr_bias_cov;
    Eigen::Matrix3d acc_bias_cov;
};
SMART_PTR(BackEnd)
