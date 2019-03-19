#pragma once
#include "util.h"
#include "backend.h"

class PLBackEnd : public BackEnd {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    PLBackEnd();
    PLBackEnd(double focal_length_,
              double gyr_n_, double acc_n_,
              double gyr_w_, double acc_w_,
              const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
              const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
              double gravity_magnitude_, int window_size_, double min_parallax_,
              double max_solver_time_in_seconds_, int max_num_iterations_,
              double cv_huber_loss_parameter_, double triangulate_default_depth_,
              double max_imu_sum_t_, int min_init_stereo_num_, int estimate_extrinsic,
              int estimate_td, double init_td);
    ~PLBackEnd();

    struct Frame : public BackEnd::Frame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual ~Frame() {}
        std::vector<uint64_t> line_id;
        Eigen::VecVector3d line_spt_n, line_ept_n; // spt: start point, ept: end point, n: normal plane
        Eigen::VecVector3d line_spt_r_n, line_ept_r_n; // r: right camera
    };
    SMART_PTR(Frame)

    struct LineFeature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        LineFeature(uint64_t feat_id_, int start_id_)
            : feat_id(feat_id_), start_id(start_id_), inv_depth{-1, -1} {}

        inline int CountNumMeas(int sw_idx) const {
            int num_meas = 0;
            for(int i = 0, n = spt_n_per_frame.size(); i < n; ++i) {
                if(start_id + i > sw_idx)
                    break;
                if(spt_r_n_per_frame[i](2) != 0)
                    num_meas += 2;
                else
                    ++num_meas;
            }
            return num_meas;
        }

        uint64_t feat_id;
        int start_id;
        double inv_depth[2]; // [0]: spt inv_depth
                             // [1]: ept inv_depth
        Eigen::DeqVector3d spt_n_per_frame, ept_n_per_frame;
        Eigen::DeqVector3d spt_r_n_per_frame, ept_r_n_per_frame;
    };
    SMART_PTR(LineFeature)

private:
    void AddFeatures(BackEnd::FramePtr frame, int& last_track_num) override;
    void SlidingWindowOld() override;
    void SlidingWindowSecondNew() override;
    int Triangulate(int sw_idx) override;
    void SolveBA() override;
    void SolveBAImu() override;
    void Marginalize() override;
    void Reset() override;
    // line feature managers
    std::map<uint64_t, LineFeature> m_lines;

    // ceres data
    void data2double() override;
    void double2data() override;
    size_t  para_lines_capacity = 1000;
    double* para_lines;
};
