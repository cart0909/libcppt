#pragma once
#include <thread>
#include <condition_variable>
#include "util.h"
#include "imu_preintegration.h"

class BackEnd {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    enum State {
        NEED_INIT,
        CV_ONLY,
        TIGHTLY
    };

    enum MarginType {
        MARGIN_SECOND_NEW,
        MARGIN_OLD
    };

    struct Frame {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
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
        std::vector<double> v_imu_timestamp;
        ImuPreintegrationPtr imu_preintegration;
    };
    SMART_PTR(Frame)

    struct Feature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        enum MapPointState {
            NEED_INIT,
            SUCCESS,
            FAIL
        };

        Feature(uint64_t feat_id_, uint sliding_window_id_)
            : feat_id(feat_id_), sliding_window_id(sliding_window_id_),
              num_meas(0), mappoint_state(NEED_INIT), inv_depth(-1.0f) {}

        uint64_t feat_id;
        uint sliding_window_id;

        uint num_meas; // num_mono * 1 + num_stereo * 2
        Eigen::DeqVector3d pt_n_per_frame;
        Eigen::DeqVector3d pt_r_n_per_frame;

        MapPointState mappoint_state;
        double inv_depth;
    };
    SMART_PTR(Feature)

    BackEnd(double focal_length_,
            double gyr_n, double acc_n,
            double gyr_w, double acc_w,
            const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
            const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
            uint window_size_ = 10);
    ~BackEnd();

    void PushFrame(FramePtr frame);

private:
    void Process();
    void ProcessFrame(FramePtr frame);
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

    std::map<uint64_t, Feature> m_features;
    uint window_size;
    uint frame_count;
    std::deque<FramePtr> v_frames;

    uint64_t next_frame_id;
    State state;
};
SMART_PTR(BackEnd)
