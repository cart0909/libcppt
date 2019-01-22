#include "backend.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include "ceres/imu_factor.h"
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>

BackEnd::BackEnd(double focal_length_,
                 double gyr_n, double acc_n,
                 double gyr_w, double acc_w,
                 const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                 const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                 int window_size_, double min_parallax_)
    : focal_length(focal_length_), p_rl(p_rl_), p_bc(p_bc_), q_rl(q_rl_), q_bc(q_bc_), window_size(window_size_),
      next_frame_id(0), state(NEED_INIT), min_parallax(min_parallax_ / focal_length), last_imu_t(-1.0f), gravity_magnitude(-1.0f)
{
    gyr_noise_cov = gyr_n * gyr_n * Eigen::Matrix3d::Identity();
    acc_noise_cov = acc_n * acc_n * Eigen::Matrix3d::Identity();
    gyr_bias_cov = gyr_w * gyr_w * Eigen::Matrix3d::Identity();
    acc_bias_cov = acc_w * acc_w * Eigen::Matrix3d::Identity();

    gyr_acc_noise_cov.setZero();
    gyr_acc_noise_cov.block<3, 3>(0, 0) = gyr_noise_cov;
    gyr_acc_noise_cov.block<3, 3>(3, 3) = acc_noise_cov;

    Eigen::Matrix6d acc_gyr_bias_cov;
    acc_gyr_bias_cov.setZero();
    acc_gyr_bias_cov.block<3, 3>(0, 0) = acc_bias_cov;
    acc_gyr_bias_cov.block<3, 3>(3, 3) = gyr_bias_cov;
    acc_gyr_bias_invcov = acc_gyr_bias_cov.inverse();

    para_pose = new double[(window_size + 1) * 7];
    para_speed_bias = new double[(window_size + 1) * 9];
    para_features = new double[para_features_capacity * 1];

    request_reset_flag = false;

    thread_ = std::thread(&BackEnd::Process, this);
}

BackEnd::~BackEnd() {
    delete [] para_pose;
    delete [] para_speed_bias;
    delete [] para_features;
}

void BackEnd::PushFrame(FramePtr frame) {
    m_buffer.lock();
    frame_buffer.emplace_back(frame);
    frame->id = next_frame_id++;
    m_buffer.unlock();
    cv_buffer.notify_one();
}

void BackEnd::Process() {
    while(1) {
        std::vector<FramePtr> measurements;
        std::unique_lock<std::mutex> lock(m_buffer);
        cv_buffer.wait(lock, [&] {
            measurements = std::vector<FramePtr>(frame_buffer.begin(), frame_buffer.end());
            frame_buffer.clear();
            return (!measurements.empty() || request_reset_flag);
        });

        if(request_reset_flag) {
            Reset();
            request_reset_flag = false;
            continue;
        }

        for(auto& frame : measurements) {
            ProcessFrame(frame);
        }
    }
}

void BackEnd::ProcessFrame(FramePtr frame) {
    // push back to sliding window
    d_frames.emplace_back(frame);

    // start the margin when the sliding window fill the frames
    marginalization_flag = AddFeaturesCheckParallax(frame);
//    LOG(INFO) << "this frame -----------------------------" << (marginalization_flag==MARGIN_OLD? "MARGIN_OLD" : "MARGIN_SECOND_NEW");

    if(state == NEED_INIT) {
        frame->q_wb = Sophus::SO3d();
        frame->p_wb.setZero();
        frame->v_wb.setZero();
        frame->ba.setZero();
        frame->bg.setZero();

        frame->imu_preintegration = std::make_shared<ImuPreintegration>(frame->bg, frame->ba, gyr_acc_noise_cov);

        int num_mps = Triangulate(0);

        if(num_mps > 50 && !frame->v_imu_timestamp.empty()) {
            last_imu_t = frame->v_imu_timestamp.back();
            state = CV_ONLY;
        }
        else {
            LOG(INFO) << "Stereo init fail, no imu info or number of mappoints less than 50: " << num_mps;
            Reset();
        }
    }
    else if(state == CV_ONLY) {
        // d_frames.size() max 11
        if(d_frames.size() >= 4) // [0, 1, 2, 3 ...
            Triangulate(d_frames.size() - 3);

        FramePtr last_frame = *(d_frames.end() - 2);

        frame->q_wb = last_frame->q_wb;
        frame->p_wb = last_frame->p_wb;
        frame->v_wb = last_frame->v_wb;
        frame->ba = last_frame->ba;
        frame->bg = last_frame->bg;

        frame->imu_preintegration = std::make_shared<ImuPreintegration>(frame->bg, frame->ba, gyr_acc_noise_cov);

        for(int i = 0, n = frame->v_acc.size(); i < n; ++i) {
            double imu_t = frame->v_imu_timestamp[i], dt = imu_t - last_imu_t;
            last_imu_t = imu_t;
            Eigen::Vector3d gyr = frame->v_gyr[i];
            Eigen::Vector3d acc = frame->v_acc[i];
            frame->imu_preintegration->push_back(dt, gyr, acc);
        }

        SolveBA();

        if(d_frames.size() == window_size + 1 && gravity_magnitude == -1.0f) {
            GyroBiasEstimation();
            if(GravityEstimation()) {
                SolveBAImu();
                exit(-1);
            }
        }

        SlidingWindow();

        if(draw_pose) {
            draw_pose(frame->id, frame->timestamp, Sophus::SE3d(frame->q_wb * q_bc, frame->q_wb * p_bc + frame->p_wb));
        }

        if(draw_mps) {
            Eigen::VecVector3d mps;

            for(auto& it : m_features) {
                auto& feat = it.second;
                if(feat.inv_depth == -1.0f)
                    continue;
                Sophus::SE3d Twb(d_frames[feat.start_id]->q_wb, d_frames[feat.start_id]->p_wb);
                Sophus::SE3d Tbc(q_bc, p_bc);
                Eigen::Vector3d x3Dc = feat.pt_n_per_frame[0] / feat.inv_depth;
                Eigen::Vector3d x3Dw = Twb * Tbc * x3Dc;
                mps.emplace_back(x3Dw);
            }

            draw_mps(mps);
        }
    }
}

BackEnd::MarginType BackEnd::AddFeaturesCheckParallax(FramePtr frame) {
    int last_track_num = 0;
    int size_frames = d_frames.size();
    for(int i = 0; i < frame->pt_id.size(); ++i) {
        auto it = m_features.find(frame->pt_id[i]);
        Feature* feat = nullptr;
        if(it == m_features.end()) {
            auto result = m_features.emplace(std::make_pair(frame->pt_id[i], Feature(frame->pt_id[i], size_frames - 1)));
            if(!result.second)
                throw std::runtime_error("m_features insert fail?");
            feat = &result.first->second;
        }
        else {
            ++last_track_num;
            feat = &it->second;
        }

        feat->pt_n_per_frame.emplace_back(frame->pt_normal_plane[i]);
        feat->pt_r_n_per_frame.emplace_back(frame->pt_r_normal_plane[i]);
    }


    if(size_frames < 2 || last_track_num < 20) {
        return MARGIN_OLD;
    }

    double parallax_sum = 0.0f;
    int parallax_num = 0;

    // check the second last frame is keyframe or not
    // parallax betwwen seconde last frame and third last frame
    // [0, 1, 2, ..., size - 3, size - 2, size - 1]

    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.start_id <= size_frames - 3 &&
           feat.start_id + static_cast<int>(feat.pt_n_per_frame.size()) - 1 >= size_frames - 2) {
            size_t idx_i = size_frames - 3 - feat.start_id;
            size_t idx_j = size_frames - 2 - feat.start_id;

            parallax_sum += (feat.pt_n_per_frame[idx_i] - feat.pt_n_per_frame[idx_j]).norm();
            parallax_num++;
        }
    }

    if(parallax_num == 0) {
        return MARGIN_OLD;
    }
    else {
        return (parallax_sum / parallax_num >= min_parallax) ? MARGIN_OLD : MARGIN_SECOND_NEW;
    }
}

void BackEnd::SlidingWindow() {
    if(marginalization_flag == MARGIN_OLD && (d_frames.size() > window_size)) {
        SlidingWindowOld();
    }
    else if(marginalization_flag == MARGIN_SECOND_NEW) {
        SlidingWindowSecondNew();
    }
}

void BackEnd::SlidingWindowOld() {
    // margin out old
    for(auto it = m_features.begin(); it != m_features.end();) {
        auto& feat = it->second;
        if(feat.start_id == 0) {
            // change parent
            if(feat.CountNumMeas(window_size) < 2) {
                it = m_features.erase(it);
            }
            else {
                if(feat.inv_depth != -1.0f) {
                    Eigen::Vector3d x3Dc0 = feat.pt_n_per_frame[0] / feat.inv_depth;
                    Eigen::Vector3d x3Db0 = q_bc * x3Dc0 + p_bc;
                    Eigen::Vector3d x3Dw = d_frames[0]->q_wb * x3Db0 + d_frames[0]->p_wb;
                    Eigen::Vector3d x3Db1 = d_frames[1]->q_wb.inverse() * (x3Dw - d_frames[1]->p_wb);
                    Eigen::Vector3d x3Dc1 = q_bc.inverse() * (x3Db1 - p_bc);
                    double inv_z1 = 1.0 / x3Dc1(2);

                    if(inv_z1 > 0) {
                        feat.inv_depth = inv_z1;
                    }
                    else {
                        feat.inv_depth = -1.0f;
                    }
                }

                feat.pt_n_per_frame.pop_front();
                feat.pt_r_n_per_frame.pop_front();

                if(feat.pt_n_per_frame.empty()) {
                    it = m_features.erase(it);
                    continue;
                }

                ++it;
            }
        }
        else {
            feat.start_id--;
            ++it;
        }
    }

    d_frames.pop_front();
}

void BackEnd::SlidingWindowSecondNew() {

    // [ 0, 1, ..., size-2, size-1]
    //  kf kf       second     new
    //              XXXXXX
    int size_frames = d_frames.size();
    for(auto it = m_features.begin(); it != m_features.end();) {
        auto& feat = it->second;

        if(feat.start_id == size_frames - 1) {
            feat.start_id--;
        }
        else {
            int j = size_frames - 2 - feat.start_id;

            if(feat.pt_n_per_frame.size() > j) {
                feat.pt_n_per_frame.erase(feat.pt_n_per_frame.begin() + j);
                feat.pt_r_n_per_frame.erase(feat.pt_r_n_per_frame.begin() + j);
            }

            if(feat.pt_n_per_frame.empty()) {
                it = m_features.erase(it);
                continue;
            }
        }
        ++it;
    }

    FramePtr second_new = *(d_frames.end() - 2);
    FramePtr latest_new = d_frames.back();

    double t0 = second_new->v_imu_timestamp.back();
    for(int i = 0, n = latest_new->v_acc.size(); i < n; ++i) {
        double dt = latest_new->v_imu_timestamp[i] - t0;
        t0 = latest_new->v_imu_timestamp[i];
        Eigen::Vector3d gyr = latest_new->v_gyr[i], acc = latest_new->v_acc[i];
        second_new->imu_preintegration->push_back(dt, gyr, acc);
    }

    latest_new->imu_preintegration = second_new->imu_preintegration;

    latest_new->v_imu_timestamp.insert(latest_new->v_imu_timestamp.begin(),
                                       second_new->v_imu_timestamp.begin(),
                                       second_new->v_imu_timestamp.end());

    latest_new->v_gyr.insert(latest_new->v_gyr.begin(),
                             second_new->v_gyr.begin(),
                             second_new->v_gyr.end());

    latest_new->v_acc.insert(latest_new->v_acc.begin(),
                             second_new->v_acc.begin(),
                             second_new->v_acc.end());

    d_frames.erase(d_frames.end() - 2);
}

int BackEnd::Triangulate(int sw_idx) {
    int num_triangulate = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;

        if(feat.inv_depth != -1.0)
            continue;

        int num_meas = feat.CountNumMeas(sw_idx);

        if(num_meas < 2)
            continue;

        Eigen::MatrixXd A(2 * num_meas, 4);
        Sophus::SE3d Tbc(q_bc, p_bc);
        Sophus::SE3d Twb0(d_frames[feat.start_id]->q_wb, d_frames[feat.start_id]->p_wb);
        Sophus::SE3d Twc0 = Twb0 * Tbc;

        int A_idx = 0;
        for(int i = 0, n = feat.pt_n_per_frame.size(); i < n; ++i) {
            int idx_i = feat.start_id + i;

            if(idx_i > sw_idx)
                break;

            Sophus::SE3d Twbi(d_frames[idx_i]->q_wb, d_frames[idx_i]->p_wb);
            Sophus::SE3d Twci = Twbi * Tbc;
            Sophus::SE3d Ti0 = Twci.inverse() * Twc0;
            Eigen::Matrix<double, 3, 4> P;
            P << Ti0.rotationMatrix(), Ti0.translation();
            A.row(A_idx++) = P.row(0) - feat.pt_n_per_frame[i](0) * P.row(2);
            A.row(A_idx++) = P.row(1) - feat.pt_n_per_frame[i](1) * P.row(2);

            if(feat.pt_r_n_per_frame[i](0) != -1) {
                Sophus::SE3d Trl(q_rl, p_rl);
                Sophus::SE3d Tri_0 = Trl * Ti0;
                P << Tri_0.rotationMatrix(), Tri_0.translation();
                A.row(A_idx++) = P.row(0) - feat.pt_r_n_per_frame[i](0) * P.row(2);
                A.row(A_idx++) = P.row(1) - feat.pt_r_n_per_frame[i](1) * P.row(2);
            }
        }

        // Solve AX = 0
        Eigen::Vector4d X = Eigen::JacobiSVD<Eigen::MatrixXd>(A, Eigen::ComputeThinV).matrixV().rightCols<1>();
        X /= X(3);

        if(X(2) < 0.1)
            continue;

        feat.inv_depth = 1.0 / X(2);
        ++num_triangulate;
    }
    return num_triangulate;
}

void BackEnd::Reset() {
    LOG(WARNING) << "BackEnd Reset...";
    d_frames.clear();
    m_features.clear();
    next_frame_id = 0;
    last_imu_t = -1.0f;
    gravity_magnitude = -1.0f;
}

void BackEnd::data2double() {
    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        size_t idx_pose = i * 7;
        size_t idx_vb = i * 9;
        std::memcpy(para_pose + idx_pose, d_frames[i]->q_wb.data(), sizeof(double) * Sophus::SO3d::num_parameters);
        std::memcpy(para_pose + idx_pose + 4, d_frames[i]->p_wb.data(), sizeof(double) * 3);
        std::memcpy(para_speed_bias + idx_vb, d_frames[i]->v_wb.data(), sizeof(double) * 3);
        std::memcpy(para_speed_bias + idx_vb + 3, d_frames[i]->ba.data(), sizeof(double) * 3);
        std::memcpy(para_speed_bias + idx_vb + 6, d_frames[i]->bg.data(), sizeof(double) * 3);
    }

    // ensure the feature array size can be filled by track features.
    if(para_features_capacity <= m_features.size()) {
        para_features_capacity *= 2;
        delete [] para_features;
        para_features = new double[para_features_capacity * 1];
    }

    size_t num_mps = 0;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.CountNumMeas(window_size) < 2 || feat.inv_depth == -1.0f) {
            continue;
        }
        para_features[num_mps++] = feat.inv_depth;
    }
}

void BackEnd::double2data() {
    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        size_t idx_pose = i * 7;
        size_t idx_vb = i * 9;
        std::memcpy(d_frames[i]->q_wb.data(), para_pose + idx_pose, sizeof(double) * Sophus::SO3d::num_parameters);
        std::memcpy(d_frames[i]->p_wb.data(), para_pose + idx_pose + 4, sizeof(double) * 3);
        std::memcpy(d_frames[i]->v_wb.data(), para_speed_bias + idx_vb , sizeof(double) * 3);
        std::memcpy(d_frames[i]->ba.data(), para_speed_bias + idx_vb + 3 , sizeof(double) * 3);
        std::memcpy(d_frames[i]->bg.data(), para_speed_bias + idx_vb + 6 , sizeof(double) * 3);
    }

    size_t num_mps = 0;
    std::vector<uint64_t> v_outlier_feat_id;
    for(auto& it : m_features) {
        auto& feat = it.second;
        if(feat.CountNumMeas(window_size) < 2 || feat.inv_depth == -1.0f) {
            continue;
        }
        feat.inv_depth = para_features[num_mps++];

        if(feat.inv_depth <= 0) {
//            LOG(INFO) << it.first << " " << feat.feat_id << " " << feat.inv_depth;
            v_outlier_feat_id.emplace_back(it.first);
        }
    }

    for(auto& it : v_outlier_feat_id) {
        m_features.erase(it);
    }
}

void BackEnd::SolveBA() {
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(5.991));
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        problem.AddParameterBlock(para_pose + i * 7, 7, local_para_se3);

        if(i == 0)
            problem.SetParameterBlockConstant(para_pose + i * 7);
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
                auto factor = new ProjectionFactor(pt_i, pt_j, q_bc, p_bc, focal_length);
                problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_features + mp_idx);
            }

            if(feat.pt_r_n_per_frame[j](0) != -1.0f) {
                Eigen::Vector3d pt_jr = feat.pt_r_n_per_frame[j];
                if(j == 0) {
                    auto factor = new SelfProjectionFactor(pt_j, pt_jr, q_rl, p_rl, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_features + mp_idx);
                }
                else {
                    auto factor = new SlaveProjectionFactor(pt_i, pt_jr, q_rl, p_rl, q_bc, p_bc, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_features + mp_idx);
                }
            }
        }
        ++mp_idx;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.num_threads = 4;
    options.max_solver_time_in_seconds = 0.05; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    double2data();
}

void BackEnd::SolveBAImu() {
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(std::sqrt(5.991));
    ceres::LocalParameterization *local_para_se3 = new LocalParameterizationSE3();

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        problem.AddParameterBlock(para_pose + i * 7, 7, local_para_se3);

        if(i == 0) {
            problem.SetParameterBlockConstant(para_pose + i * 7);
        }
        else {
            auto factor = new ImuFactor(d_frames[i]->imu_preintegration,
                                        Eigen::Vector3d(0, 0, -gravity_magnitude),
                                        acc_gyr_bias_invcov);
            problem.AddResidualBlock(factor, NULL,
                                     para_pose + (i - 1) * 7,
                                     para_speed_bias + (i - 1) * 9,
                                     para_pose + i * 7,
                                     para_speed_bias + i * 9);
        }
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
                auto factor = new ProjectionFactor(pt_i, pt_j, q_bc, p_bc, focal_length);
                problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_features + mp_idx);
            }

            if(feat.pt_r_n_per_frame[j](0) != -1.0f) {
                Eigen::Vector3d pt_jr = feat.pt_r_n_per_frame[j];
                if(j == 0) {
                    auto factor = new SelfProjectionFactor(pt_j, pt_jr, q_rl, p_rl, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_features + mp_idx);
                }
                else {
                    auto factor = new SlaveProjectionFactor(pt_i, pt_jr, q_rl, p_rl, q_bc, p_bc, focal_length);
                    problem.AddResidualBlock(factor, loss_function, para_pose + id_i * 7, para_pose + id_j * 7, para_features + mp_idx);
                }
            }
        }
        ++mp_idx;
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.num_threads = 4;
//    options.max_solver_time_in_seconds = 0.05; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport();
    double2data();
}

void BackEnd::SolvePnP(FramePtr frame) {
    // pnp this help debug
    std::vector<cv::Point2f> v_pts;
    std::vector<cv::Point3d> v_mps;

    for(int i = 0, n = frame->pt.size(); i < n; ++i) {
        auto it = m_features.find(frame->pt_id[i]);
        if(it != m_features.end()) {
            if(it->second.inv_depth != -1.0f) {
                Eigen::Vector3d x3Dc = it->second.pt_n_per_frame[0] / it->second.inv_depth;
                Eigen::Vector3d x3Db = q_bc * x3Dc + p_bc;
                Eigen::Vector3d x3Dw = d_frames[it->second.start_id]->q_wb * x3Db + d_frames[it->second.start_id]->p_wb;
                v_mps.emplace_back(x3Dw(0), x3Dw(1), x3Dw(2));
                v_pts.emplace_back(frame->pt_normal_plane[i](0), frame->pt_normal_plane[i](1));
            }
        }
    }

    cv::Mat rvec, tvec;
    cv::solvePnPRansac(v_mps, v_pts, cv::Mat::eye(3, 3, CV_64F), cv::noArray(),
                       rvec, tvec, false, 100, 8.0/focal_length, 0.99);
    cv::Mat R;
    cv::Rodrigues(rvec, R);
    Eigen::Matrix3d RR;
    cv::cv2eigen(R, RR);
    Eigen::Quaterniond q(RR);
    q.normalize();

    Sophus::SO3d q_cw(q);
    Eigen::Vector3d tcw;
    cv::cv2eigen(tvec, tcw);

    Sophus::SO3d q_bw = q_bc * q_cw;
    Eigen::Vector3d p_bw = q_bc * tcw + p_bc;

    frame->q_wb = q_bw.inverse();
    frame->p_wb = -(q_bw.inverse() * p_bw);
}

bool BackEnd::GyroBiasEstimation() {
    double err_0 = 0, err_1 = 0;

    Eigen::Matrix3d H = Eigen::Matrix3d::Zero();
    Eigen::Vector3d b = Eigen::Vector3d::Zero();

    for(int i = 0, n = d_frames.size(); i < n - 1; ++i) {
        FramePtr frame_i = d_frames[i];
        FramePtr frame_i1 = d_frames[i + 1];
        ImuPreintegrationPtr imu_preintegration = frame_i1->imu_preintegration;
        Eigen::Matrix3d JR_bg = imu_preintegration->mJR_bg;
        Sophus::SO3d delRi_i1 = imu_preintegration->mdelRij;
        Sophus::SO3d qwi = frame_i->q_wb, qwi1 = frame_i1->q_wb;
        Eigen::Vector3d residual = (delRi_i1.inverse() * (qwi1.inverse() * qwi)).log();
        err_0 += residual.squaredNorm();
        H += JR_bg.transpose() * JR_bg;
        b += JR_bg.transpose() * residual;
    }

    Eigen::Vector3d bg = d_frames[0]->bg + H.ldlt().solve(b);

    for(auto& frame : d_frames) {
        frame->imu_preintegration->Repropagate(bg, Eigen::Vector3d::Zero());
        frame->bg = bg;
    }

    for(int i = 0, n = d_frames.size(); i < n - 1; ++i) {
        FramePtr frame_i = d_frames[i];
        FramePtr frame_i1 = d_frames[i + 1];
        Sophus::SO3d delRi_i1 = frame_i1->imu_preintegration->mdelRij;
        Sophus::SO3d qwi = frame_i->q_wb, qwi1 = frame_i1->q_wb;
        Eigen::Vector3d residual = (delRi_i1.inverse() * (qwi1.inverse() * qwi)).log();
        err_1 += residual.squaredNorm();
    }

    LOG(INFO) << "step 1, bg: " << bg(0) << " " << bg(1) << " " << bg(2);

    return true;
}

bool BackEnd::ScaleAndGravityApproximation() {

    Eigen::MatrixXd A(3*(d_frames.size() - 2), 4);
    Eigen::VectorXd b(3*(d_frames.size() - 2));

    Sophus::SO3d q_cb = q_bc.inverse();
    Eigen::Vector3d p_cb = -(q_cb * p_bc);

    for(int i = 0, n = d_frames.size(); i < n - 2; ++i) {
        FramePtr frame0 = d_frames[i], frame1 = d_frames[i + 1], frame2 = d_frames[i + 2];
        // FIXME
        Eigen::Vector3d pwc0 = frame0->q_wb * p_bc + frame0->p_wb,
                        pwc1 = frame1->q_wb * p_bc + frame1->p_wb,
                        pwc2 = frame2->q_wb * p_bc + frame2->p_wb;
        Sophus::SO3d qwb0 = frame0->q_wb,
                     qwb1 = frame1->q_wb,
                     qwb2 = frame2->q_wb,
                     qwc0 = qwb0 * q_bc,
                     qwc1 = qwb1 * q_bc,
                     qwc2 = qwb2 * q_bc;
        //
        Eigen::Vector3d dp01 = frame1->imu_preintegration->mdelPij,
                        dp12 = frame2->imu_preintegration->mdelPij,
                        dv01 = frame1->imu_preintegration->mdelVij;
        double dt01 = frame1->imu_preintegration->mdel_tij,
               dt12 = frame2->imu_preintegration->mdel_tij;
        Eigen::Vector3d lambda = (pwc1 - pwc0) * dt12 - (pwc2 - pwc1) * dt01;
        Eigen::Matrix3d beta = 0.5 * Eigen::Matrix3d::Identity() * dt01 * dt12 * (dt01 + dt12);
        Eigen::Vector3d gamma = (qwc0.matrix() - qwc1.matrix()) * p_cb * dt12
                               -(qwc1.matrix() - qwc2.matrix()) * p_cb * dt01
                               + qwb0 * dp01 * dt12 - qwb1 * dp12 * dt01
                               - qwb0 * dv01 * dt01 * dt12;

        A.block<3, 1>(i * 3, 0) = lambda;
        A.block<3, 3>(i * 3, 1) = beta;
        b.segment(i * 3, 3) = gamma;
    }

    // TODO
    Eigen::Vector4d x = A.colPivHouseholderQr().solve(b);
    double scale_value = x(0);
    Eigen::Vector3d gw = x.tail(3);

    LOG(INFO) << "step 2, scale: " << scale_value << " gw: " << gw(0) << " " << gw(1) << " " << gw(2);

    AccBiasEstimationWithScaleAndGravityRefinement(gw);

    return true;
}

bool BackEnd::AccBiasEstimationWithScaleAndGravityRefinement(const Eigen::Vector3d& gw) {
    Eigen::Vector3d gw_dir = gw.normalized();
    Eigen::Vector3d gI_dir(0, 0, -1);
    double G = 9.81;

    Eigen::Vector3d v_dir = gI_dir.cross(gw_dir);
    v_dir.normalize();
    double theta = std::atan2(gI_dir.cross(gw_dir).norm(), gI_dir.dot(gw_dir));
    Sophus::SO3d RwI = Sophus::SO3d::exp(theta * v_dir);

    // update law
    // RwI <- RwI * Exp(d_Theta)
    // d_Thera = [dtx, dty, 0]'

    // wanted
    // gw = RwI*G*gI_dir
    Eigen::MatrixXd A(3 * (d_frames.size() - 2), 6);
    Eigen::VectorXd b(3 * (d_frames.size() - 2));

    Sophus::SO3d q_cb = q_bc.inverse();
    Eigen::Vector3d p_cb = -(q_cb * p_bc);


    for(int i = 0, n = d_frames.size(); i < n - 2; ++i) {
        FramePtr frame0 = d_frames[i], frame1 = d_frames[i + 1], frame2 = d_frames[i + 2];
        // FIXME
        Eigen::Vector3d pwc0 = frame0->q_wb * p_bc + frame0->p_wb,
                        pwc1 = frame1->q_wb * p_bc + frame1->p_wb,
                        pwc2 = frame2->q_wb * p_bc + frame2->p_wb;
        Sophus::SO3d qwb0 = frame0->q_wb,
                     qwb1 = frame1->q_wb,
                     qwb2 = frame2->q_wb,
                     qwc0 = qwb0 * q_bc,
                     qwc1 = qwb1 * q_bc,
                     qwc2 = qwb2 * q_bc;
        //
        Eigen::Vector3d dp01 = frame1->imu_preintegration->mdelPij,
                        dp12 = frame2->imu_preintegration->mdelPij,
                        dv01 = frame1->imu_preintegration->mdelVij;
        double dt01 = frame1->imu_preintegration->mdel_tij,
               dt12 = frame2->imu_preintegration->mdel_tij;
        Eigen::Matrix3d JP01_ba = frame1->imu_preintegration->mJP_ba,
                        JP12_ba = frame2->imu_preintegration->mJP_ba,
                        JV01_ba = frame1->imu_preintegration->mJV_ba;

        Eigen::Vector3d lambda = (pwc1 - pwc0) * dt12 - (pwc2 - pwc1) * dt01;
        Eigen::Matrix3d phi = -0.5 * RwI.matrix() * Sophus::SO3d::hat(gI_dir) * G * (dt01 * dt12) * (dt01 + dt12);
        Eigen::Matrix3d zeta = qwb1.matrix() * JP12_ba * dt01
                             + qwb0.matrix() * JV01_ba * dt01 * dt12
                             - qwb0.matrix() * JV01_ba * dt12;
        Eigen::Vector3d psi = qwb0 * dp01 * dt12 - qwb1 * dp12 * dt01 - qwb0 * dv01 * dt01 * dt12
                           + (qwc0.matrix() - qwc1.matrix()) * p_cb * dt12
                           - (qwc1.matrix() - qwc2.matrix()) * p_cb * dt01
                           - 0.5 * (RwI * gI_dir) * G * (dt01 * dt12) * (dt01 + dt12);

        A.block<3, 1>(3 * i, 0) = lambda;
        A.block<3, 2>(3 * i, 1) = phi.leftCols(2);
        A.block<3, 3>(3 * i, 3) = zeta;
        b.segment(3 * i, 3) = psi;
    }

    Eigen::Vector6d x = A.colPivHouseholderQr().solve(b);

    double scale_value = x(0);
    Eigen::Vector3d delta_theta(x(1), x(2), 0);
    Eigen::Vector3d ba = x.tail(3);

    LOG(INFO) << "step 3, scale: " << scale_value << " delta_theta: " << delta_theta(0) << " " << delta_theta(1)
              << " ba: " << ba(0) << " " << ba(1) << " " << ba(2);

    return true;
}

bool BackEnd::GravityEstimation() {
    Eigen::MatrixXd A(3*(d_frames.size() - 2), 3);
    Eigen::VectorXd b(3*(d_frames.size() - 2));
    Sophus::SO3d q_cb = q_bc.inverse();
    Eigen::Vector3d p_cb = -(q_cb * p_bc);
    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();

    for(int i = 0, n = d_frames.size(); i < n - 2; ++i) {
        FramePtr frame0 = d_frames[i], frame1 = d_frames[i + 1], frame2 = d_frames[i + 2];
        Eigen::Vector3d pwc0 = frame0->q_wb * p_bc + frame0->p_wb,
                        pwc1 = frame1->q_wb * p_bc + frame1->p_wb,
                        pwc2 = frame2->q_wb * p_bc + frame2->p_wb;
        Sophus::SO3d qwb0 = frame0->q_wb,
                     qwb1 = frame1->q_wb,
                     qwb2 = frame2->q_wb,
                     qwc0 = qwb0 * q_bc,
                     qwc1 = qwb1 * q_bc,
                     qwc2 = qwb2 * q_bc;
        Eigen::Vector3d dp01 = frame1->imu_preintegration->mdelPij,
                        dp12 = frame2->imu_preintegration->mdelPij,
                        dv01 = frame1->imu_preintegration->mdelVij;
        double dt01 = frame1->imu_preintegration->mdel_tij,
               dt12 = frame2->imu_preintegration->mdel_tij;

        Eigen::Matrix3d beta = 0.5 * I3x3 * (dt01 * dt12) * (dt01 + dt12);
        Eigen::Vector3d alpha = (pwc2 - pwc1) * dt01 - (pwc1 - pwc0) * dt12
                              + (qwc0.matrix() - qwc1.matrix()) * p_cb * dt12
                              - (qwc1.matrix() - qwc2.matrix()) * p_cb * dt01
                              + qwb0 * dp01 * dt12 - qwb1 * dp12 * dt01
                              - qwb0 * dv01 * dt01 * dt12;

        A.block<3, 3>(3 * i, 0) = beta;
        b.segment(3 * i, 3) = alpha;
    }

    Eigen::Vector3d gw = A.colPivHouseholderQr().solve(b);

    LOG(INFO) << "step 2, gw: " << gw(0) << " " << gw(1) << " " << gw(2) << ", |gw|: " << gw.norm();

    Eigen::Vector3d gw_dir = gw.normalized();
    Eigen::Vector3d gI_dir(0, 0, -1);
    double G = gw.norm();

    if(G < 9.7f || G > 9.9f)
        return false;

    gravity_magnitude = G;

    Eigen::Vector3d v_dir = gI_dir.cross(gw_dir);
    v_dir.normalize();
    double theta = std::atan2(gI_dir.cross(gw_dir).norm(), gI_dir.dot(gw_dir));
    Sophus::SO3d RwI = Sophus::SO3d::exp(theta * v_dir);
    Sophus::SO3d RIw = RwI.inverse();

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        d_frames[i]->q_wb = RIw * d_frames[i]->q_wb;
        d_frames[i]->p_wb = RIw * d_frames[i]->p_wb;
    }

    return true;
}
