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
        Eigen::VecVector3d spt_n, ept_n; // spt: start point, ept: end point, n: normal plane
        Eigen::VecVector3d spt_r_n, ept_r_n; // r: right camera
    };
    SMART_PTR(Frame)

    struct LineFeature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t feat_id;
        int start_id;
        double inv_depth[2]; // [0]: spt inv_depth
                             // [1]: ept inv_depth
        Eigen::DeqVector3d spt_n_per_frame, ept_n_per_frame;
        Eigen::DeqVector3d spt_r_n_per_frame, ept_r_n_per_frame;
    };
    SMART_PTR(LineFeature)

private:
    // ceres data
    size_t  para_line_features_capacity = 1000;
    double* para_line_features[2];
};
