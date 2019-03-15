#include "pl_backend.h"

PLBackEnd::PLBackEnd() {}

PLBackEnd::PLBackEnd(double focal_length_,
                     double gyr_n_, double acc_n_,
                     double gyr_w_, double acc_w_,
                     const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                     const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                     double gravity_magnitude_, int window_size_, double min_parallax_,
                     double max_solver_time_in_seconds_, int max_num_iterations_,
                     double cv_huber_loss_parameter_, double triangulate_default_depth_,
                     double max_imu_sum_t_, int min_init_stereo_num_, int estimate_extrinsic,
                     int estimate_td, double init_td)
    : BackEnd(focal_length_, gyr_n_, acc_n_, gyr_w_, acc_w_, p_rl_, p_bc_, q_rl_, q_bc_, gravity_magnitude_,
              window_size_, min_parallax_, max_solver_time_in_seconds_, max_num_iterations_,
              cv_huber_loss_parameter_, triangulate_default_depth_, max_imu_sum_t_, min_init_stereo_num_,
              estimate_extrinsic, estimate_td, init_td)
{
    para_line_features[0] = new double[para_line_features_capacity];
    para_line_features[1] = new double[para_line_features_capacity];
}

PLBackEnd::~PLBackEnd() {
    delete [] para_line_features[0];
    delete [] para_line_features[1];
}
