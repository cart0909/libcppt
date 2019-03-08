#pragma once
#include "backend.h"

class RGBDBackEnd : public BackEnd {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RGBDBackEnd(double focal_length_,
                double gyr_n_, double acc_n_,
                double gyr_w_, double acc_w_,
                const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                double gravity_magnitude_, int window_size_, double min_parallax_,
                double max_solver_time_in_seconds_, int max_num_iterations_,
                double cv_huber_loss_parameter_, double triangulate_default_depth_,
                double max_imu_sum_t_, int min_init_stereo_num_, int estimate_extrinsic,
                int estimate_td, double init_td);
private:
    void SolveBAImu() override;
};
SMART_PTR(RGBDBackEnd)
