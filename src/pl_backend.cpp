#include "pl_backend.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/local_parameterization_line.h"
#include "ceres/projection_factor.h"
#include "ceres/projection_td_factor.h"
#include "ceres/line_projection_factor.h"
#include "vins/imu_factor.h"
#define USE_POINT 1

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
    para_lines = new double[para_lines_capacity * Plucker::Line3d::num_parameters];
}

PLBackEnd::~PLBackEnd() {
    delete [] para_lines;
}

void PLBackEnd::AddFeatures(BackEnd::FramePtr frame_, int& last_track_num) {
    BackEnd::AddFeatures(frame_, last_track_num);
    FramePtr frame = std::dynamic_pointer_cast<Frame>(frame_);
    int frame_idx = d_frames.size() - 1;

    for(int i = 0; i < frame->line_id.size(); ++i) {
        auto it = m_lines.find(frame->line_id[i]);
        if(it == m_lines.end()) {
            auto result = m_lines.emplace(std::make_pair(frame->line_id[i],
                                                         LineFeature(frame->line_id[i], frame_idx)));
            if(!result.second)
                throw std::runtime_error("m_lines insert fail?");
            LineFeature& line = result.first->second;
            line.spt_n_per_frame.emplace_back(frame->line_spt_n[i]);
            line.ept_n_per_frame.emplace_back(frame->line_ept_n[i]);
            line.spt_r_n_per_frame.emplace_back(frame->line_spt_r_n[i]);
            line.ept_r_n_per_frame.emplace_back(frame->line_ept_r_n[i]);
        }
        else {
            LineFeature& line = it->second;
            line.spt_n_per_frame.emplace_back(frame->line_spt_n[i]);
            line.ept_n_per_frame.emplace_back(frame->line_ept_n[i]);
            line.spt_r_n_per_frame.emplace_back(frame->line_spt_r_n[i]);
            line.ept_r_n_per_frame.emplace_back(frame->line_ept_r_n[i]);
        }
    }
}

void PLBackEnd::SlidingWindowOld() {
    for(auto it = m_lines.begin(); it != m_lines.end();) {
        auto& line = it->second;
        if(line.start_id == 0) {
            if(line.CountNumMeas(window_size) < 2) {
                it = m_lines.erase(it);
            }
            else {
                line.spt_n_per_frame.pop_front();
                line.ept_n_per_frame.pop_front();
                line.spt_r_n_per_frame.pop_front();
                line.ept_r_n_per_frame.pop_front();

                if(line.spt_n_per_frame.empty()) {
                    it = m_lines.erase(it);
                    continue;
                }
                ++it;
            }
        }
        else {
            --line.start_id;
            ++it;
        }
    }

    BackEnd::SlidingWindowOld();
}

void PLBackEnd::SlidingWindowSecondNew() {
    int size_frames = d_frames.size();
    for(auto it = m_lines.begin(); it != m_lines.end();) {
        auto& line = it->second;

        if(line.start_id == size_frames - 1) {
            line.start_id--;
        }
        else {
            int j = size_frames - 2 - line.start_id;

            if(line.spt_n_per_frame.size() > j) {
                line.spt_n_per_frame.erase(line.spt_n_per_frame.begin() + j);
                line.ept_n_per_frame.erase(line.ept_n_per_frame.begin() + j);
                line.spt_r_n_per_frame.erase(line.spt_r_n_per_frame.begin() + j);
                line.ept_r_n_per_frame.erase(line.ept_r_n_per_frame.begin() + j);
            }

            if(line.spt_n_per_frame.empty()) {
                it = m_lines.erase(it);
                continue;
            }
        }
        ++it;
    }
    BackEnd::SlidingWindowSecondNew();
}

int PLBackEnd::Triangulate(int sw_idx) {
    for(auto& it : m_lines) {
        auto& line = it.second;

        if(!line.need_triangulate)
            continue;

        int num_meas = line.CountNumMeas(sw_idx);

        if(num_meas < 3)
            continue;

        Eigen::MatrixXd A(2 * num_meas, 6);
        Sophus::SE3d Tbc(q_bc, p_bc);
        Sophus::SE3d Twb0(d_frames[line.start_id]->q_wb, d_frames[line.start_id]->p_wb);
        Sophus::SE3d Twc0 = Twb0 * Tbc;

        int A_idx = 0;
        for(int i = 0, n = line.spt_n_per_frame.size(); i < n; ++i) {
            int idx_i = line.start_id + i;

            if(idx_i > sw_idx)
                break;

            Sophus::SE3d Twbi(d_frames[idx_i]->q_wb, d_frames[idx_i]->p_wb);
            Sophus::SE3d Twci = Twbi * Tbc;
            Sophus::SE3d Ti0 = Twci.inverse() * Twc0;
            Eigen::Matrix<double, 3, 6> Pi0;
            Pi0 << Ti0.rotationMatrix(), Sophus::SO3d::hat(Ti0.translation()) * Ti0.rotationMatrix();
            A.row(A_idx++) = line.spt_n_per_frame[i].transpose() * Pi0;
            A.row(A_idx++) = line.ept_n_per_frame[i].transpose() * Pi0;

            if(line.spt_r_n_per_frame[i](2) != 0) {
                Sophus::SE3d Trl(q_rl, p_rl);
                Sophus::SE3d Tri_0 = Trl * Ti0;
                Pi0 << Tri_0.rotationMatrix(), Sophus::SO3d::hat(Tri_0.translation()) * Tri_0.rotationMatrix();
                A.row(A_idx++) = line.spt_r_n_per_frame[i].transpose() * Pi0;
                A.row(A_idx++) = line.ept_r_n_per_frame[i].transpose() * Pi0;
            }
        }

        Eigen::Vector6d x = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        Eigen::Vector3d n, dir;
        Plucker::Correction::LMPC(x.head<3>(), x.tail<3>(), n, dir);
        Plucker::Line3d L0(dir, n, Plucker::PLUCKER_L_M);
        Plucker::Line3d P_line0(Eigen::Vector3d(0, 0, 0), line.spt_n_per_frame[0], Plucker::TWO_POINT),
                        Q_line0(Eigen::Vector3d(0, 0, 0), line.ept_n_per_frame[0], Plucker::TWO_POINT);

        Eigen::Vector3d p1_star, q1_star;
        Plucker::Feet(P_line0, L0, p1_star);
        Plucker::Feet(Q_line0, L0, q1_star);

        if(p1_star(2) < 0 || q1_star(2) < 0 || (p1_star - q1_star).norm() > 5)
            continue;

        Eigen::Vector3d l = L0.m() / L0.m().head<2>().norm();
        Eigen::Vector2d residuals;
        residuals << focal_length * l.dot(line.spt_n_per_frame[0]),
                     focal_length * l.dot(line.ept_n_per_frame[0]);

        if(residuals.norm() > 3)
            continue;

        line.Lw = Twc0 * L0;
        line.need_triangulate = false;
    }

    return BackEnd::Triangulate(sw_idx);
}

void PLBackEnd::data2double() {
    BackEnd::data2double();
    // ensure the feature array size can be filled by track features.
    if(para_lines_capacity <= m_lines.size()) {
        para_lines_capacity *= 2;
        delete [] para_lines;
        para_lines = new double[para_lines_capacity * Plucker::Line3d::num_parameters];
    }

    size_t num_lines = 0;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.need_triangulate)
            continue;
        Eigen::Matrix3d Q;
        double w1, w2;
        line.Lw.Orthonormal(Q, w1, w2);
        Sophus::SO3d U(Q);
        Sophus::SO2d W(w1, w2);
        std::memcpy(para_lines + num_lines * 6, U.data(), sizeof(double) * 4);
        std::memcpy(para_lines + num_lines * 6 + 4, W.data(), sizeof(double) * 2);
        ++num_lines;
    }
}

void PLBackEnd::double2data() {
    Sophus::SO3d q_w0b0 = d_frames[0]->q_wb;
    Eigen::Vector3d p_w0b0 = d_frames[0]->p_wb;

    Sophus::SO3d q_w1b0;
    Eigen::Vector3d p_w1b0;
    std::memcpy(q_w1b0.data(), para_pose, sizeof(double) * 4);
    std::memcpy(p_w1b0.data(), para_pose + 4, sizeof(double) * 3);
    double y_diff = Sophus::R2ypr(q_w0b0 * q_w1b0.inverse())(0);
    Sophus::SO3d q_w0w1 = Sophus::ypr2R<double>(y_diff, 0, 0);
    Eigen::Vector3d p_w0w1 = -(q_w0w1 * p_w1b0) + p_w0b0;
    Sophus::SE3d Tw0w1(q_w0w1, p_w0w1);

    size_t num_lines = 0;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.need_triangulate)
            continue;
        Sophus::SO3d U;
        Sophus::SO2d W;
        std::memcpy(U.data(), para_lines + num_lines * 6, sizeof(double) * 4);
        std::memcpy(W.data(), para_lines + num_lines * 6 + 4, sizeof(double) * 2);
        Plucker::Line3d Lw1;
        Lw1.FromOrthonormal(U.matrix(), W.unit_complex()(0), W.unit_complex()(1));
        line.Lw = Tw0w1 * Lw1;
        ++num_lines;
    }

    BackEnd::double2data();

    // remove line outliers after backend::double2data finished
    std::vector<uint64_t> v_outlier_line_id;
    Sophus::SE3d Tbc(q_bc, p_bc), Trl(q_rl, p_rl);
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.need_triangulate)
            continue;

        double ave_r_norm = 0.0;
        int count = 0;
        bool is_outlier = false;
        for(int i = 0, n = line.spt_n_per_frame.size(); i < n; ++i) {
            int id = line.start_id + i;
            Sophus::SE3d Twb(d_frames[id]->q_wb, d_frames[id]->p_wb);
            Sophus::SE3d Tcw = (Twb * Tbc).inverse();
            Plucker::Line3d Lc = Tcw * line.Lw;

            Plucker::Line3d P_line0(Eigen::Vector3d(0, 0, 0), line.spt_n_per_frame[i], Plucker::TWO_POINT),
                            Q_line0(Eigen::Vector3d(0, 0, 0), line.ept_n_per_frame[i], Plucker::TWO_POINT);

            Eigen::Vector3d p1_star, q1_star;
            Plucker::Feet(P_line0, Lc, p1_star);
            Plucker::Feet(Q_line0, Lc, q1_star);

            if(p1_star(2) < 0|| q1_star(2) < 0 || (p1_star - q1_star).norm() > 5) {
                v_outlier_line_id.emplace_back(it.first);
                is_outlier = true;
                break;
            }

            Eigen::Vector3d l = Lc.m() / Lc.m().head<2>().norm();
            Eigen::Vector2d residuals;
            residuals << focal_length * l.dot(line.spt_n_per_frame[i]),
                         focal_length * l.dot(line.ept_n_per_frame[i]);

            ave_r_norm += residuals.norm();
            ++count;
        }

        if(!is_outlier) {
            ave_r_norm /= count;
            if(ave_r_norm > 3) {
                v_outlier_line_id.emplace_back(it.first);
            }
        }
    }

    for(auto& it : v_outlier_line_id)
        m_lines.erase(it);
}

void PLBackEnd::SolveBA() {
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3(),
                                 *local_para_so3xso2 = new LocalParameterizationSO3xSO2();

    problem.AddParameterBlock(para_ex_bc, 7, local_para_se3);
    problem.AddParameterBlock(para_ex_sm, 7, local_para_se3);
    problem.SetParameterBlockConstant(para_ex_bc);
    problem.SetParameterBlockConstant(para_ex_sm);

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        problem.AddParameterBlock(para_pose + i * 7, 7, local_para_se3);

        if(i == 0)
            problem.SetParameterBlockConstant(para_pose);
    }

    size_t mp_idx = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.CountNumMeas(window_size) < 2 || feat.inv_depth == -1.0f) {
            continue;
        }

        size_t id_i = feat.start_id;

        Eigen::Vector3d pt_i = feat.pt_n_per_frame[0];
        for(int j = 0, n = feat.pt_n_per_frame.size(); j < n; ++j) {
            size_t id_j = id_i + j;
            Eigen::Vector3d pt_j = feat.pt_n_per_frame[j];

            if(j != 0) {
                auto factor = new ProjectionExFactor(pt_i, pt_j, focal_length);
                problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_features + mp_idx);
            }

            if(feat.pt_r_n_per_frame[j](2) != 0) {
                Eigen::Vector3d pt_jr = feat.pt_r_n_per_frame[j];
                if(j == 0) {
                    auto factor = new SelfProjectionExFactor(pt_j, pt_jr, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_ex_sm, para_features + mp_idx);
                }
                else {
                    auto factor = new SlaveProjectionExFactor(pt_i, pt_jr, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_features + mp_idx);
                }
            }
        }
        ++mp_idx;
    }

    size_t line_idx = 0;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.need_triangulate) {
            continue;
        }

        problem.AddParameterBlock(para_lines + line_idx * 6, 6, local_para_so3xso2);

        for(int i = 0, n = line.spt_n_per_frame.size(); i < n; ++i) {
            int id = line.start_id + i;
            Eigen::Vector3d spt = line.spt_n_per_frame[i], ept = line.ept_n_per_frame[i];
            auto factor = new LineProjectionFactor(spt, ept, focal_length);
            problem.AddResidualBlock(factor, loss_function, para_pose + id * 7, para_ex_bc, para_lines + line_idx * 6);

            if(line.spt_r_n_per_frame[i](2) != 0) {
                Eigen::Vector3d spt_r = line.spt_r_n_per_frame[i], ept_r = line.ept_r_n_per_frame[i];
                auto factor = new LineSlaveProjectionFactor(spt_r, ept_r, focal_length);
                problem.AddResidualBlock(factor, loss_function, para_pose + id * 7, para_ex_bc, para_ex_sm, para_lines + line_idx * 6);
            }
        }
        ++line_idx;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = max_num_iterations;
    options.num_threads = 8;
//    options.max_solver_time_in_seconds = max_solver_time_in_seconds; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
//    LOG(INFO) << summary.FullReport();
    double2data();
}

void PLBackEnd::SolveBAImu() {
    Tracer::TraceBegin("BA");
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3(),
                                 *local_para_so3xso2 = new LocalParameterizationSO3xSO2();

    problem.AddParameterBlock(para_ex_bc, 7, local_para_se3);
    problem.AddParameterBlock(para_ex_sm, 7, local_para_se3);

    if(!enable_estimate_extrinsic || d_frames.back()->v_wb.norm() < 0.2) {
        problem.SetParameterBlockConstant(para_ex_bc);
        problem.SetParameterBlockConstant(para_ex_sm);
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

#if USE_POINT
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
#endif

    size_t line_idx = 0;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.need_triangulate) {
            continue;
        }

        problem.AddParameterBlock(para_lines + line_idx * 6, 6, local_para_so3xso2);

        for(int i = 0, n = line.spt_n_per_frame.size(); i < n; ++i) {
            int id = line.start_id + i;
            Eigen::Vector3d spt = line.spt_n_per_frame[i], ept = line.ept_n_per_frame[i];
            auto factor = new LineProjectionFactor(spt, ept, focal_length);
            problem.AddResidualBlock(factor, loss_function, para_pose + id * 7, para_ex_bc, para_lines + line_idx * 6);

            if(line.spt_r_n_per_frame[i](2) != 0) {
                Eigen::Vector3d spt_r = line.spt_r_n_per_frame[i], ept_r = line.ept_r_n_per_frame[i];
                auto factor = new LineSlaveProjectionFactor(spt_r, ept_r, focal_length);
                problem.AddResidualBlock(factor, loss_function, para_pose + id * 7, para_ex_bc, para_ex_sm, para_lines + line_idx * 6);
            }
        }
        ++line_idx;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
    options.max_num_iterations = max_num_iterations;
    options.num_threads = 8;
    options.max_solver_time_in_seconds = max_solver_time_in_seconds; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.FullReport() << std::endl;
    double2data();

    for(int i = 1, n = d_frames.size(); i < n; ++i)
        d_frames[i]->imupreinte->repropagate(d_frames[i-1]->ba, d_frames[i-1]->bg);
    Tracer::TraceEnd();
    Marginalize();
}

void PLBackEnd::Marginalize() {
    if(marginalization_flag == MARGIN_OLD) {
        ScopedTrace st("margin_old");
        margin_mps.clear();
        auto loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
        auto line_loss_function = new ceres::HuberLoss(0.3);
        auto margin_info = new MarginalizationInfo();
        data2double();

        if(last_margin_info) {
            std::vector<int> drop_set;
            for(int i = 0, n = para_margin_block.size(); i < n ; ++i) {
                if(para_margin_block[i] == para_pose ||
                   para_margin_block[i] == para_speed_bias)
                    drop_set.emplace_back(i);
            }

            auto factor = new MarginalizationFactor(last_margin_info);
            auto residual_block_info = new ResidualBlockInfo(factor, NULL,
                                                             para_margin_block,
                                                             drop_set);
            margin_info->addResidualBlockInfo(residual_block_info);
        }

        // imu
        if(d_frames[1]->imupreinte->sum_dt < max_imu_sum_t) {
            auto factor = new IMUFactor(d_frames[1]->imupreinte, gw);
            auto residual_block_info = new ResidualBlockInfo(factor, NULL,
                                                             std::vector<double*>{para_pose, para_speed_bias,
                                                                                  para_pose + 7, para_speed_bias + 9},
                                                             std::vector<int>{0, 1});
            margin_info->addResidualBlockInfo(residual_block_info);
        }
#if USE_POINT
        // features
        int feature_index = -1;
        for(auto& it : m_features) {
            auto& feat = it.second;

            if(feat.inv_depth == -1.0f)
                continue;

            if(feat.CountNumMeas(window_size) < 2) {
                // for ui debug
                Eigen::Vector3d pt_i = feat.pt_n_per_frame[0];
                Eigen::Vector3d x3Dc = pt_i / feat.inv_depth;
                Eigen::Vector3d x3Db = q_bc * x3Dc + p_bc;
                Eigen::Vector3d x3Dw = d_frames[0]->q_wb * x3Db + d_frames[0]->p_wb;
                margin_mps.emplace_back(x3Dw);
                continue;
            }
            ++feature_index;

            int id_i = feat.start_id;
            if(id_i != 0)
                continue;

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
                        auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                         std::vector<double*>{para_pose, para_pose + id_j * 7, para_ex_bc, para_features + feature_index, para_Td},
                                                                         std::vector<int>{0, 3});
                        margin_info->addResidualBlockInfo(residual_block_info);
                    }
                    else {
                        auto factor = new ProjectionExFactor(pt_i, pt_j, focal_length);
                        auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                         std::vector<double*>{para_pose, para_pose + id_j * 7, para_ex_bc, para_features + feature_index},
                                                                         std::vector<int>{0, 3});
                        margin_info->addResidualBlockInfo(residual_block_info);
                    }
                }

                if(feat.pt_r_n_per_frame[j](2) != 0) {
                    Eigen::Vector3d pt_jr = feat.pt_r_n_per_frame[j];
                    Eigen::Vector3d velocity_jr = feat.velocity_r_per_frame[j];
                    if(j == 0) {
                        if(estimate_time_delay) {
                             auto factor = new SelfProjectionTdFactor(pt_j, velocity_j, pt_jr, velocity_jr, d_frames[id_j]->td, focal_length);
                             auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                              std::vector<double*>{para_ex_sm, para_features + feature_index, para_Td},
                                                                              std::vector<int>{1});
                             margin_info->addResidualBlockInfo(residual_block_info);
                        }
                        else {
                            auto factor = new SelfProjectionExFactor(pt_j, pt_jr, focal_length);
                            auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                             std::vector<double*>{para_ex_sm, para_features + feature_index},
                                                                             std::vector<int>{1});
                            margin_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                    else {
                        if(estimate_time_delay) {
                            auto factor = new SlaveProjectionTdFactor(pt_i, velocity_i, d_frames[id_i]->td,
                                                                      pt_jr, velocity_jr, d_frames[id_j]->td,
                                                                      focal_length);
                            auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                             std::vector<double*>{para_pose, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_features + feature_index, para_Td},
                                                                             std::vector<int>{0, 4});
                                                        margin_info->addResidualBlockInfo(residual_block_info);
                        }
                        else {
                            auto factor = new SlaveProjectionExFactor(pt_i, pt_jr, focal_length);
                            auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                             std::vector<double*>{para_pose, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_features + feature_index},
                                                                             std::vector<int>{0, 4});
                            margin_info->addResidualBlockInfo(residual_block_info);
                        }
                    }
                }
            }
        }
#endif
        // line
        size_t line_idx = -1;
        for(auto& it : m_lines) {
            auto& line = it.second;
            if(line.CountNumMeas(window_size) < 2 || line.need_triangulate) {
                continue;
            }

            line_idx++;

            if(line.start_id != 0)
                continue;

            for(int i = 0, n = line.spt_n_per_frame.size(); i < n; ++i) {
                Eigen::Vector3d spt = line.spt_n_per_frame[i], ept = line.ept_n_per_frame[i];
                auto factor = new LineProjectionFactor(spt, ept, focal_length);
                if(i == 0) {
                    auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                     std::vector<double*>{para_pose, para_ex_bc, para_lines + line_idx * 6},
                                                                     std::vector<int>{0, 2});
                    margin_info->addResidualBlockInfo(residual_block_info);
                }
                else {
                    auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                     std::vector<double*>{para_pose + i * 7, para_ex_bc, para_lines + line_idx * 6},
                                                                     std::vector<int>{2});
                    margin_info->addResidualBlockInfo(residual_block_info);
                }

                if(line.spt_r_n_per_frame[i](2) != 0) {
                    Eigen::Vector3d spt_r = line.spt_r_n_per_frame[i], ept_r = line.ept_r_n_per_frame[i];
                    auto factor = new LineSlaveProjectionFactor(spt_r, ept_r, focal_length);
                    if(i == 0) {
                        auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                         std::vector<double*>{para_pose, para_ex_bc, para_ex_sm, para_lines + line_idx * 6},
                                                                         std::vector<int>{0, 3});
                        margin_info->addResidualBlockInfo(residual_block_info);
                    }
                    else {
                        auto residual_block_info = new ResidualBlockInfo(factor, loss_function,
                                                                         std::vector<double*>{para_pose + i * 7, para_ex_bc, para_ex_sm, para_lines + line_idx * 6},
                                                                         std::vector<int>{3});
                        margin_info->addResidualBlockInfo(residual_block_info);
                    }
                }
            }
        }

        margin_info->preMarginalize();
        margin_info->marginalize();

        std::map<long, double *> addr_shift;
        for(int i = 1, n = d_frames.size(); i < n; ++i) {
            addr_shift[reinterpret_cast<long>(para_pose + 7 * i)] = para_pose + 7 * (i - 1);
            addr_shift[reinterpret_cast<long>(para_speed_bias + 9 * i)] = para_speed_bias + 9 * (i - 1);
        }

        addr_shift[reinterpret_cast<long>(para_ex_bc)] = para_ex_bc;
        addr_shift[reinterpret_cast<long>(para_ex_sm)] = para_ex_sm;

        if(estimate_time_delay) {
            addr_shift[reinterpret_cast<long>(para_Td)] = para_Td;
        }

        std::vector<double*> parameter_blocks = margin_info->getParameterBlocks(addr_shift);


        if(last_margin_info)
            delete last_margin_info;
        last_margin_info = margin_info;
        para_margin_block = parameter_blocks;
    }
    else if(marginalization_flag == MARGIN_SECOND_NEW) {
        ScopedTrace st("margin_2nd");
        auto it = std::find(para_margin_block.begin(), para_margin_block.end(), para_pose + 7 * (d_frames.size() - 2));
        if(last_margin_info && it != para_margin_block.end()) {
            auto margin_info = new MarginalizationInfo();
            data2double();

            std::vector<int> drop_set;
            drop_set.emplace_back(it - para_margin_block.begin());
            auto factor = new MarginalizationFactor(last_margin_info);
            auto residual_block_info = new ResidualBlockInfo(factor, NULL,
                                                             para_margin_block,
                                                             drop_set);
            margin_info->addResidualBlockInfo(residual_block_info);

            margin_info->preMarginalize();
            margin_info->marginalize();

            std::map<long, double*> addr_shift;
            for(int i = 0, n = d_frames.size(); i < n; ++i) {
                if(i == n - 2) {
                    continue;
                }
                else if(i == n - 1) {
                    addr_shift[reinterpret_cast<long>(para_pose + 7 * i)] = para_pose + 7 * (i - 1);
                    addr_shift[reinterpret_cast<long>(para_speed_bias + 9 * i)] = para_speed_bias + 9 * (i - 1);
                }
                else {
                    addr_shift[reinterpret_cast<long>(para_pose + 7 * i)] = para_pose + 7 * i;
                    addr_shift[reinterpret_cast<long>(para_speed_bias + 9 * i)] = para_speed_bias + 9 * i;
                }
            }

            addr_shift[reinterpret_cast<long>(para_ex_bc)] = para_ex_bc;
            addr_shift[reinterpret_cast<long>(para_ex_sm)] = para_ex_sm;

            if(estimate_time_delay) {
                addr_shift[reinterpret_cast<long>(para_Td)] = para_Td;
            }

            std::vector<double*> parameter_blocks = margin_info->getParameterBlocks(addr_shift);

            if(last_margin_info)
                delete last_margin_info;
            last_margin_info = margin_info;
            para_margin_block = parameter_blocks;
        }
    }
}

void PLBackEnd::Reset() {
    BackEnd::Reset();
    m_lines.clear();
}

void PLBackEnd::Publish() {
    BackEnd::Publish();

    if(!pub_lines.empty()) {
        Eigen::VecVector3d v_Pw, v_Qw;
        Sophus::SE3d Tbc(q_bc, p_bc);
        for(auto& it : m_lines) {
            auto& line = it.second;
            if(line.CountNumMeas(window_size) < 2 || line.need_triangulate) {
                continue;
            }

            int id_i = line.start_id;
            Sophus::SE3d Twb(d_frames[id_i]->q_wb, d_frames[id_i]->p_wb), Twc = Twb * Tbc;
            Plucker::Line3d P_line0(Eigen::Vector3d(0, 0, 0), line.spt_n_per_frame[0], Plucker::TWO_POINT),
                            Q_line0(Eigen::Vector3d(0, 0, 0), line.ept_n_per_frame[0], Plucker::TWO_POINT);

            Plucker::Line3d Lc = Twc.inverse() * line.Lw;
            Eigen::Vector3d p1_star, q1_star;
            Plucker::Feet(P_line0, Lc, p1_star);
            Plucker::Feet(Q_line0, Lc, q1_star);

            Eigen::Vector3d Pw, Qw;
            Pw = Twc * p1_star;
            Qw = Twc * q1_star;

            v_Pw.emplace_back(Pw);
            v_Qw.emplace_back(Qw);
        }

        for(auto& pub : pub_lines) {
            pub(v_Pw, v_Qw);
        }
    }
}

void PLBackEnd::PredictNextFramePose(BackEnd::FramePtr ref_frame, BackEnd::FramePtr cur_frame) {
    BackEnd::PredictNextFramePose(ref_frame, cur_frame);
//    cur_frame->q_wb = ref_frame->q_wb;
//    cur_frame->p_wb = ref_frame->p_wb;
//    cur_frame->v_wb = ref_frame->v_wb;
//    cur_frame->ba = ref_frame->ba;
//    cur_frame->bg = ref_frame->bg;
//    cur_frame->imupreinte = std::make_shared<IntegrationBase>(
//                ref_frame->v_acc.back(), ref_frame->v_gyr.back(),
//                ref_frame->ba, ref_frame->bg,
//                acc_n, gyr_n, acc_w, gyr_w);

//    Eigen::Vector3d gyr_0 = ref_frame->v_gyr.back(), acc_0 = ref_frame->v_acc.back();
//    double t0 = ref_frame->v_imu_timestamp.back();

//    for(int i = 0, n = cur_frame->v_acc.size(); i < n; ++i) {
//        double t = cur_frame->v_imu_timestamp[i], dt = t - t0;
//        Eigen::Vector3d gyr = cur_frame->v_gyr[i];
//        Eigen::Vector3d acc = cur_frame->v_acc[i];
//        cur_frame->imupreinte->push_back(dt, acc, gyr);

//        gyr_0 = gyr;
//        acc_0 = acc;
//        t0 = t;
//    }
}

#undef USE_POINT
