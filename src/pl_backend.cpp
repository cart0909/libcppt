#include "pl_backend.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include "ceres/projection_td_factor.h"
#include "ceres/line_projection_factor.h"
#include "vins/imu_factor.h"

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
    para_lines = new double[para_lines_capacity * 2];
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
                if(line.inv_depth[0] != -1.0f) {
                    Eigen::Vector3d Pc0 = line.spt_n_per_frame[0] / line.inv_depth[0];
                    Eigen::Vector3d Pb0 = q_bc * Pc0 + p_bc;
                    Eigen::Vector3d Pw = d_frames[0]->q_wb * Pb0 + d_frames[0]->p_wb;
                    Eigen::Vector3d Pb1 = d_frames[1]->q_wb.inverse() * (Pw - d_frames[1]->p_wb);
                    Eigen::Vector3d Pc1 = q_bc.inverse() * (Pb1 - p_bc);
                    double inv_z1 = 1.0f / Pc1(2);

                    if(inv_z1 > 0) {
                        line.inv_depth[0] = inv_z1;
                    }
                    else {
                        line.inv_depth[0] = -1.0f;
                    }
                }

                if(line.inv_depth[1] != -1.0f) {
                    Eigen::Vector3d Qc0 = line.ept_n_per_frame[0] / line.inv_depth[1];
                    Eigen::Vector3d Qb0 = q_bc * Qc0 + p_bc;
                    Eigen::Vector3d Qw = d_frames[0]->q_wb * Qb0 + d_frames[0]->p_wb;
                    Eigen::Vector3d Qb1 = d_frames[1]->q_wb.inverse() * (Qw - d_frames[1]->p_wb);
                    Eigen::Vector3d Qc1 = q_bc.inverse() * (Qb1 - p_bc);
                    double inv_z1 = 1.0f / Qc1(2);

                    if(inv_z1 > 0) {
                        line.inv_depth[1] = inv_z1;
                    }
                    else {
                        line.inv_depth[1] = -1.0f;
                    }
                }

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

        if(line.inv_depth[0] != -1.0f && line.inv_depth[1] != -1.0f)
            continue;

        int num_meas = line.CountNumMeas(sw_idx);

        if(num_meas < 2)
            continue;

        Eigen::MatrixXd Ap(2 * num_meas, 4), Aq(2 * num_meas, 4);
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
            Eigen::Matrix<double, 3, 4> P;
            P << Ti0.rotationMatrix(), Ti0.translation();
            Ap.row(A_idx)   = P.row(0) - line.spt_n_per_frame[i](0) * P.row(2);
            Aq.row(A_idx++) = P.row(0) - line.ept_n_per_frame[i](0) * P.row(2);
            Ap.row(A_idx)   = P.row(1) - line.spt_n_per_frame[i](1) * P.row(2);
            Aq.row(A_idx++) = P.row(1) - line.ept_n_per_frame[i](1) * P.row(2);

            if(line.spt_r_n_per_frame[i](2) != 0) {
                Sophus::SE3d Trl(q_rl, p_rl);
                Sophus::SE3d Tri_0 = Trl * Ti0;
                P << Tri_0.rotationMatrix(), Tri_0.translation();
                Ap.row(A_idx)   = P.row(0) - line.spt_r_n_per_frame[i](0) * P.row(2);
                Aq.row(A_idx++) = P.row(0) - line.ept_r_n_per_frame[i](0) * P.row(2);
                Ap.row(A_idx)   = P.row(1) - line.spt_r_n_per_frame[i](1) * P.row(2);
                Aq.row(A_idx++) = P.row(1) - line.ept_r_n_per_frame[i](1) * P.row(2);
            }
        }

        // Solve AX=0
        Eigen::Vector4d P = Eigen::JacobiSVD<Eigen::MatrixXd>(Ap, Eigen::ComputeThinV).matrixV().rightCols<1>(),
                        Q = Eigen::JacobiSVD<Eigen::MatrixXd>(Aq, Eigen::ComputeThinV).matrixV().rightCols<1>();

        P /= P(3);
        Q /= Q(3);

        if(P(2) < 0.1) {
            line.inv_depth[0] = 1.0f / triangulate_default_depth;
        }
        else {
            line.inv_depth[0] = 1.0f / P(2);
        }

        if(Q(2) < 0.1) {
            line.inv_depth[1] = 1.0f / triangulate_default_depth;
        }
        else {
            line.inv_depth[1] = 1.0f / Q(2);
        }
    }

    return BackEnd::Triangulate(sw_idx);
}

void PLBackEnd::data2double() {
    BackEnd::data2double();
    // ensure the feature array size can be filled by track features.
    if(para_lines_capacity <= m_lines.size()) {
        para_lines_capacity *= 2;
        delete [] para_lines;
        para_lines = new double[para_lines_capacity * 2];
    }

    size_t num_lines = 0;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.inv_depth[0] == -1.0f || line.inv_depth[1] == -1.0f)
            continue;
        std::memcpy(para_lines + num_lines * 2, line.inv_depth, sizeof(double) * 2);
        ++num_lines;
    }
}

void PLBackEnd::double2data() {
    BackEnd::double2data();

    size_t num_lines = 0;
    std::vector<uint64_t> v_outlier_line_id;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.inv_depth[0] == -1.0f || line.inv_depth[1] == -1.0f)
            continue;
        std::memcpy(line.inv_depth, para_lines + (num_lines++) * 2, sizeof(double) * 2);

        if(line.inv_depth[0] <= 0 || line.inv_depth[1] <= 0) {
            v_outlier_line_id.emplace_back(it.first);
        }
        //else {
            // TODO: remove the heigh cost lines
        //}
    }

    for(auto& it : v_outlier_line_id)
        m_lines.equal_range(it);
}

void PLBackEnd::SolveBA() {
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

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
    cv::Mat draw(480, 640, CV_8UC3, cv::Scalar::all(0));
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

            if(id_j == window_size) {
                Eigen::Vector2d p = pt_j.head<2>() * focal_length;
                p(0) += 320;
                p(1) += 240;
                cv::circle(draw, cv::Point(p(0), p(1)), 1, cv::Scalar(0, 255, 0), -1);
            }

            if(j != 0 && (id_i + n - 1) == window_size) {
                Eigen::Vector2d p = pt_j.head<2>() * focal_length;
                p(0) += 320;
                p(1) += 240;
                Eigen::Vector2d q = feat.pt_n_per_frame[j-1].head<2>() * focal_length;
                q(0) += 320;
                q(1) += 240;
                cv::line(draw, cv::Point(p(0), p(1)), cv::Point(q(0), q(1)), cv::Scalar(0, 255, 255), 1);
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
        if(line.CountNumMeas(window_size) < 2 || line.inv_depth[0] == -1.0f || line.inv_depth[1] == -1.0f) {
            continue;
        }

        size_t id_i = line.start_id;
        Eigen::Vector3d spt_i = line.spt_n_per_frame[0], ept_i = line.ept_n_per_frame[0];
        for(int j = 0, n = line.spt_n_per_frame.size(); j < n; ++j) {
            size_t id_j = id_i + j;
            Eigen::Vector3d spt_j = line.spt_n_per_frame[j], ept_j = line.ept_n_per_frame[j];

            if(j != 0) {
//                auto factor = new LineProjectionFactor(spt_i, ept_i, spt_j, ept_j, focal_length);
//                problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_lines + line_idx * 2);
            }

            if(id_j == window_size) {
                Eigen::Vector2d p = spt_j.head<2>() * focal_length;
                Eigen::Vector2d q = ept_j.head<2>() * focal_length;
                p(0) += 320;
                p(1) += 240;
                q(0) += 320;
                q(1) += 240;
                cv::line(draw, cv::Point(p(0), p(1)), cv::Point(q(0), q(1)), cv::Scalar(255, 255, 255), 1);
                cv::circle(draw, cv::Point(p(0), p(1)), 2, cv::Scalar(0, 0, 255), -1);
                cv::circle(draw, cv::Point(q(0), q(1)), 2, cv::Scalar(255, 0, 0), -1);
            }

            if(j != 0 && (id_i + n - 1) == window_size) {
                Eigen::Vector2d p0 = spt_j.head<2>() * focal_length;
                Eigen::Vector2d q0 = ept_j.head<2>() * focal_length;
                p0(0) += 320;
                p0(1) += 240;
                q0(0) += 320;
                q0(1) += 240;
                Eigen::Vector2d p1 = line.spt_n_per_frame[j-1].head<2>() * focal_length;
                Eigen::Vector2d q1 = line.ept_n_per_frame[j-1].head<2>() * focal_length;
                p1(0) += 320;
                p1(1) += 240;
                q1(0) += 320;
                q1(1) += 240;
                cv::line(draw, cv::Point(p0(0), p0(1)), cv::Point(p1(0), p1(1)), cv::Scalar(0, 0, 255), 1);
                cv::line(draw, cv::Point(q0(0), q0(1)), cv::Point(q1(0), q1(1)), cv::Scalar(255, 0, 0), 1);
            }

            if(line.spt_r_n_per_frame[j](2) != 0) {
                Eigen::Vector3d spt_jr = line.spt_r_n_per_frame[j], ept_jr = line.ept_r_n_per_frame[j];
                if(j == 0) {
//                    auto factor = new LineSelfProjectionFactor(spt_j, ept_j, spt_jr, ept_jr, focal_length);
//                    problem.AddResidualBlock(factor, loss_function, para_ex_sm, para_lines + line_idx * 2);
                }
                else {
//                    auto factor = new LineSlaveProjectionFactor(spt_i, ept_i, spt_jr, ept_jr, focal_length);
//                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_lines + line_idx * 2);
                }
            }
        }
        ++line_idx;
    }

    cv::imshow("draw", draw);
    cv::waitKey(1);

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
}

void PLBackEnd::SolveBAImu() {
    Tracer::TraceBegin("BA");
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

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

    size_t mp_idx = 0;
    cv::Mat draw(480, 640, CV_8UC3, cv::Scalar::all(0));
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

//            if(id_j == window_size) {
//                Eigen::Vector2d p = pt_j.head<2>() * focal_length;
//                p(0) += 320;
//                p(1) += 240;
//                cv::circle(draw, cv::Point(p(0), p(1)), 1, cv::Scalar(0, 255, 0), -1);
//            }

//            if(j != 0 && (id_i + n - 1) == window_size) {
//                Eigen::Vector2d p = pt_j.head<2>() * focal_length;
//                p(0) += 320;
//                p(1) += 240;
//                Eigen::Vector2d q = feat.pt_n_per_frame[j-1].head<2>() * focal_length;
//                q(0) += 320;
//                q(1) += 240;
//                cv::line(draw, cv::Point(p(0), p(1)), cv::Point(q(0), q(1)), cv::Scalar(0, 255, 255), 1);
//            }

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

    size_t line_idx = 0;
    for(auto& it : m_lines) {
        auto& line = it.second;
        if(line.CountNumMeas(window_size) < 2 || line.inv_depth[0] == -1.0f || line.inv_depth[1] == -1.0f) {
            continue;
        }

        size_t id_i = line.start_id;
        Eigen::Vector3d spt_i = line.spt_n_per_frame[0], ept_i = line.ept_n_per_frame[0];
        for(int j = 0, n = line.spt_n_per_frame.size(); j < n; ++j) {
            size_t id_j = id_i + j;
            Eigen::Vector3d spt_j = line.spt_n_per_frame[j], ept_j = line.ept_n_per_frame[j];
            if(j != 0) {
//                auto factor = new LineProjectionFactor(spt_i, ept_i, spt_j, ept_j, focal_length);
//                problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_lines + line_idx * 2);
            }

            if(id_j == window_size) {
                Eigen::Vector2d p = spt_j.head<2>() * focal_length;
                Eigen::Vector2d q = ept_j.head<2>() * focal_length;
                p(0) += 320;
                p(1) += 240;
                q(0) += 320;
                q(1) += 240;
                cv::line(draw, cv::Point(p(0), p(1)), cv::Point(q(0), q(1)), cv::Scalar(255, 255, 255), 1);
                cv::circle(draw, cv::Point(p(0), p(1)), 2, cv::Scalar(0, 0, 255), -1);
                cv::circle(draw, cv::Point(q(0), q(1)), 2, cv::Scalar(255, 0, 0), -1);
            }

            if(j != 0 && (id_i + n - 1) == window_size) {
                Eigen::Vector2d p0 = spt_j.head<2>() * focal_length;
                Eigen::Vector2d q0 = ept_j.head<2>() * focal_length;
                p0(0) += 320;
                p0(1) += 240;
                q0(0) += 320;
                q0(1) += 240;
                Eigen::Vector2d p1 = line.spt_n_per_frame[j-1].head<2>() * focal_length;
                Eigen::Vector2d q1 = line.ept_n_per_frame[j-1].head<2>() * focal_length;
                p1(0) += 320;
                p1(1) += 240;
                q1(0) += 320;
                q1(1) += 240;
                cv::line(draw, cv::Point(p0(0), p0(1)), cv::Point(p1(0), p1(1)), cv::Scalar(0, 0, 255), 1);
                cv::line(draw, cv::Point(q0(0), q0(1)), cv::Point(q1(0), q1(1)), cv::Scalar(255, 0, 0), 1);
            }

            if(line.spt_r_n_per_frame[j](2) != 0) {
                Eigen::Vector3d spt_jr = line.spt_r_n_per_frame[j], ept_jr = line.ept_r_n_per_frame[j];
                if(j == 0) {
//                    auto factor = new LineSelfProjectionFactor(spt_j, ept_j, spt_jr, ept_jr, focal_length);
//                    problem.AddResidualBlock(factor, loss_function, para_ex_sm, para_lines + line_idx * 2);
                }
                else {
//                    auto factor = new LineSlaveProjectionFactor(spt_i, ept_i, spt_jr, ept_jr, focal_length);
//                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_ex_bc, para_ex_sm, para_lines + line_idx * 2);
                }
            }
        }
        ++line_idx;
    }

    cv::imshow("draw", draw);
    cv::waitKey(1);

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

void PLBackEnd::Marginalize() {
    if(marginalization_flag == MARGIN_OLD) {
        ScopedTrace st("margin_old");
        margin_mps.clear();
        auto loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
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