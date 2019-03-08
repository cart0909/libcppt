#include "rgbd_backend.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include "ceres/projection_td_factor.h"
#include "vins/imu_factor.h"

RGBDBackEnd::RGBDBackEnd(double focal_length_,
                         double gyr_n_, double acc_n_,
                         double gyr_w_, double acc_w_,
                         const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                         const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                         double gravity_magnitude_, int window_size_, double min_parallax_,
                         double max_solver_time_in_seconds_, int max_num_iterations_,
                         double cv_huber_loss_parameter_, double triangulate_default_depth_,
                         double max_imu_sum_t_, int min_init_stereo_num_, int estimate_extrinsic,
                         int estimate_td, double init_td)
    : BackEnd(focal_length_,
              gyr_n_, acc_n_,
              gyr_w_, acc_w_,
              p_rl_, p_bc_,
              q_rl_, q_bc_,
              gravity_magnitude_, window_size_, min_parallax_,
              max_solver_time_in_seconds_, max_num_iterations_,
              cv_huber_loss_parameter_, triangulate_default_depth_,
              max_imu_sum_t_, min_init_stereo_num_, estimate_extrinsic,
              estimate_td, init_td)
{}

void RGBDBackEnd::SolveBAImu() {
    Tracer::TraceBegin("BA");
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

    problem.AddParameterBlock(para_ex_bc, 7, local_para_se3);
    problem.AddParameterBlock(para_ex_sm, 7, local_para_se3);
    problem.SetParameterBlockConstant(para_ex_sm);

    if(!enable_estimate_extrinsic || d_frames.back()->v_wb.norm() < 0.2) {
        problem.SetParameterBlockConstant(para_ex_bc);
    }

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        problem.AddParameterBlock(para_pose + i * 7, 7, local_para_se3);

        if(i != 0 && d_frames[i]->imupreinte->sum_dt < max_imu_sum_t) {
            auto factor = new IMUFactor(d_frames[i]->imupreinte, gw);
            problem.AddResidualBlock(factor, NULL,
                                     para_pose + (i - 1) * 7,
                                     para_speed_bias + (i - 1) * 9,
                                     para_pose + i * 7,
                                     para_speed_bias + i * 9);
        }
    }

    if(last_margin_info) {
        auto factor = new MarginalizationFactor(last_margin_info);
        problem.AddResidualBlock(factor, NULL, para_margin_block);
    }

    size_t mp_idx = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.CountNumMeas(window_size) < 2 || feat.inv_depth == -1.0f) {
            continue;
        }

        size_t id_i = feat.start_id;
        Eigen::Vector3d pt_i = feat.pt_n_per_frame[0];
        Eigen::Vector3d velocity_i = feat.velocity_per_frame[0];
        for(int j = 0, n = feat.pt_n_per_frame.size(); j < n; ++j) {
            size_t id_j = id_i + j;
            Eigen::Vector3d pt_j = feat.pt_n_per_frame[j];
            Eigen::Vector3d velocity_j = feat.velocity_per_frame[j];

            if(j != 0) {
                if(estimate_time_delay) {
                    auto factor = new ProjectionTdFactor(pt_i, velocity_i, d_frames[id_i]->td,
                                                         pt_j, velocity_j, d_frames[id_j]->td,
                                                         focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_features + mp_idx, para_Td);
                }
                else {
                    auto factor = new ProjectionExFactor(pt_i, pt_j, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_features + mp_idx);
                }
            }

            if(feat.pt_r_n_per_frame[j](2) != 0) {
                Eigen::Vector3d pt_jr = feat.pt_r_n_per_frame[j];
                Eigen::Vector3d velocity_jr = feat.velocity_r_per_frame[j];
                if(j == 0) {
                    if(estimate_time_delay) {
                        auto factor = new SelfProjectionTdFactor(pt_j, velocity_j, pt_jr, velocity_jr, d_frames[id_j]->td, focal_length);
                        problem.AddResidualBlock(factor, loss_function, para_ex_sm, para_features + mp_idx, para_Td);
                    }
                    else {
                        auto factor = new SelfProjectionExFactor(pt_j, pt_jr, focal_length);
                        problem.AddResidualBlock(factor, loss_function, para_ex_sm, para_features + mp_idx);
                    }
                }
                else {
                    if(estimate_time_delay) {
                        auto factor = new SlaveProjectionTdFactor(pt_i, velocity_i, d_frames[id_i]->td,
                                                                  pt_jr, velocity_jr, d_frames[id_j]->td,
                                                                  focal_length);
                        problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_features + mp_idx, para_Td);
                    }
                    else {
                        auto factor = new SlaveProjectionExFactor(pt_i, pt_jr, focal_length);
                        problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_features + mp_idx);
                    }
                }
            }
        }
        ++mp_idx;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = max_num_iterations;
    options.num_threads = 1;
    options.max_solver_time_in_seconds = max_solver_time_in_seconds; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
//    LOG(INFO) << summary.FullReport();
    double2data();

    for(int i = 1, n = d_frames.size(); i < n; ++i)
        d_frames[i]->imupreinte->repropagate(d_frames[i-1]->ba, d_frames[i-1]->bg);
    Tracer::TraceEnd();
    Marginalize();
}
