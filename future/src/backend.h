#pragma once
#include <thread>
#include <atomic>
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

        Feature(uint64_t feat_id_, int start_id_)
            : feat_id(feat_id_), start_id(start_id_),
              mappoint_state(NEED_INIT), inv_depth(-1.0f) {}

        uint64_t feat_id;
        int start_id;

        inline int CountNumMeas(int sw_idx) const {
            int num_meas = 0;
            for(int i = 0, n = pt_n_per_frame.size(); i < n; ++i) {
                if(start_id + i > sw_idx)
                    break;
                if(pt_r_n_per_frame[i](0) != -1.0)
                    num_meas += 2;
                else
                    num_meas += 1;
            }
            return num_meas;
        }

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
            int window_size_ = 10, double min_parallax_ = 10.0f);
    ~BackEnd();

    void PushFrame(FramePtr frame);

    inline void SetDrawMapPointCallback(std::function<void(const Eigen::VecVector3d&)> callback) {
        draw_mps = callback;
    }

    inline void SetDrawPoseCallback(std::function<void(uint64_t, double, const Sophus::SE3d&)> callback) {
        draw_pose = callback;
    }

    inline void ResetRequest() {
        request_reset_flag = true;
    }

private:
    void Process();
    void ProcessFrame(FramePtr frame);
    MarginType AddFeaturesCheckParallax(FramePtr frame);
    void SlidingWindow();
    void SlidingWindowOld();
    void SlidingWindowSecondNew();
    int Triangulate(int sw_idx);
    void Reset();
    void SolveBA();

    std::thread thread_;
    std::mutex  m_buffer;
    std::condition_variable cv_buffer;
    std::deque<FramePtr> frame_buffer; // [ 0,  1, ..., 8 ,         9 |  10] size 11
                                       //  kf  kf      kf  second new   new
    double focal_length;
    Eigen::Vector3d p_rl, p_bc;
    Sophus::SO3d    q_rl, q_bc;

    Eigen::Matrix3d gyr_noise_cov;
    Eigen::Matrix3d acc_noise_cov;
    Eigen::Matrix3d gyr_bias_cov;
    Eigen::Matrix3d acc_bias_cov;

    std::map<uint64_t, Feature> m_features;
    int window_size;
    std::deque<FramePtr> d_frames;

    uint64_t next_frame_id;
    State state;

    double min_parallax;
    MarginType marginalization_flag;

    std::function<void(const Eigen::VecVector3d&)> draw_mps;

    // ceres data
    void data2double();
    void double2data();
    double* para_pose; // Twb
    double* para_speed_bias; // vwb bg ba
    size_t  para_features_capacity = 1000;
    double* para_features; // inv_z

    // maintain system
    std::atomic<bool> request_reset_flag;

    std::function<void(uint64_t, double, const Sophus::SE3d&)> draw_pose;
};
SMART_PTR(BackEnd)
