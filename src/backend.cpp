#include "backend.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include "ceres/projection_td_factor.h"
#include "vins/imu_factor.h"
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include "add_msg/ImuPredict.h"
#include "geometry_msgs/Pose.h"
#include "ceres/lidar_factor.h"
#define LIDARFACTOR 0
BackEnd::BackEnd() {}

BackEnd::BackEnd(double focal_length_,
                 double gyr_n_, double acc_n_,
                 double gyr_w_, double acc_w_,
                 const Eigen::Vector3d& p_rl_, const Eigen::Vector3d& p_bc_,
                 const Sophus::SO3d& q_rl_, const Sophus::SO3d& q_bc_,
                 double gravity_magnitude_, int window_size_, double min_parallax_,
                 double max_solver_time_in_seconds_, int max_num_iterations_,
                 double cv_huber_loss_parameter_, double triangulate_default_depth_,
                 double max_imu_sum_t_, int min_init_stereo_num_, int estimate_extrinsic,
                 int estimate_td, double init_td, int use_lidar_tracking_)
    : focal_length(focal_length_), p_rl(p_rl_), p_bc(p_bc_), q_rl(q_rl_), q_bc(q_bc_),
      gyr_n(gyr_n_), acc_n(acc_n_), gyr_w(gyr_w_), acc_w(acc_w_), window_size(window_size_),
      next_frame_id(0), state(NEED_INIT), min_parallax(min_parallax_ / focal_length), last_imu_t(-1.0f),
      gravity_magnitude(gravity_magnitude_), gw(0, 0, gravity_magnitude_), last_margin_info(nullptr),
      max_solver_time_in_seconds(max_solver_time_in_seconds_), max_num_iterations(max_num_iterations_),
      cv_huber_loss_parameter(cv_huber_loss_parameter_), triangulate_default_depth(triangulate_default_depth_),
      max_imu_sum_t(max_imu_sum_t_), min_init_stereo_num(min_init_stereo_num_), estimate_time_delay(estimate_td),
      time_delay(init_td), use_lidar_tracking(use_lidar_tracking_)
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
    //Set body to lidar extrinsic of parameter
    Eigen::Vector3d lidar_p_body;
    lidar_p_body.z() = -0.05;
    Tlb.translation() = lidar_p_body;
    Tlb.setQuaternion(Eigen::Quaterniond(1,0,0,0));
    enable_estimate_extrinsic = estimate_extrinsic;
}

BackEnd::~BackEnd() {
    delete [] para_pose;
    delete [] para_speed_bias;
    delete [] para_features;
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
        imuinfo_tmp.BackTime = -10;
        int num_mps = Triangulate(0);

        if(num_mps >= min_init_stereo_num && frame->v_imu_timestamp.size() >= 5) {
            frame->q_wb = InitFirstIMUPose(frame->v_acc);
            frame->imupreinte = std::make_shared<IntegrationBase>(frame->v_acc[0], frame->v_gyr[0], frame->ba, frame->bg, acc_n, gyr_n, acc_w, gyr_w);
            last_imu_t = frame->v_imu_timestamp.back();
            state = CV_ONLY;
        }
        else {
            LOG(INFO) << "Stereo init fail, no imu info or number of mappoints less than " <<  min_init_stereo_num  << ": " << num_mps;
            Reset();
        }
    }
    else {
        // d_frames.size() max 11
        if(d_frames.size() >= 4) // [0, 1, 2, 3 ...
            Triangulate(d_frames.size() - 3);

        FramePtr last_frame = *(d_frames.end() - 2);
        // predict next frame pose from last frame
        PredictNextFramePose(last_frame, frame);

        if(state == CV_ONLY)
            SolveBA();
        else if(state == TIGHTLY)
            SolveBAImu();

        if(d_frames.size() == window_size + 1 && state == CV_ONLY) {
            GyroBiasEstimation();
            SolveBAImu();
            state = TIGHTLY;
        }

        if(state == TIGHTLY) {
            SlidingWindow();
            Publish();
            //DrawUI();
        }
    }
}


void BackEnd::ProcessFrame(FramePtr frame,
                           std::pair<sensor_msgs::PointCloud2ConstPtr, sensor_msgs::PointCloud2ConstPtr> lidarInfo,
                           bool &start_this_frame, Eigen::VecVector3d& lidar_acc, Eigen::VecVector3d& lidar_gyro, std::vector<double>& lidar_imut){
    // push back to sliding window
    d_frames.emplace_back(frame);

    // start the margin when the sliding window fill the frames
    marginalization_flag = AddFeaturesCheckParallax(frame);
    //LOG(INFO) << "this frame -----------------------------" << (marginalization_flag==MARGIN_OLD? "MARGIN_OLD" : "MARGIN_SECOND_NEW");

    if(state == NEED_INIT) {
        frame->q_wb = Sophus::SO3d();
        frame->p_wb.setZero();
        frame->v_wb.setZero();
        frame->ba.setZero();
        frame->bg.setZero();
        imuinfo_tmp.BackTime = -10;
        int num_mps = Triangulate(0);

        if(num_mps >= min_init_stereo_num && frame->v_imu_timestamp.size() >= 5) {
            frame->q_wb = InitFirstIMUPose(frame->v_acc);
            frame->imupreinte = std::make_shared<IntegrationBase>(frame->v_acc[0], frame->v_gyr[0], frame->ba, frame->bg, acc_n, gyr_n, acc_w, gyr_w);
            last_imu_t = frame->v_imu_timestamp.back();
            state = CV_ONLY;
        }
        else {
            LOG(INFO) << "Stereo init fail, no imu info or number of mappoints less than " <<  min_init_stereo_num  << ": " << num_mps;
            Reset();
        }
    }
    else {
        // d_frames.size() max 11
        if(d_frames.size() >= 4) // [0, 1, 2, 3 ...
            Triangulate(d_frames.size() - 3);

        if(start_this_frame){
            frame->lidar_time = lidarInfo.first->header.stamp.toSec();
        }

        FramePtr last_frame = *(d_frames.end() - 2);
        // predict next frame pose from last frame
        PredictNextFramePose(last_frame, frame);
        if(lidar_acc.size() > 0)
            PredictNextLidarPose(frame, lidar_acc, lidar_gyro, lidar_imut);

        TransLidarPointToImage(frame, lidarInfo);

        if(state == CV_ONLY)
            SolveBA();
        else if(state == TIGHTLY)
            SolveBAImu();

        if(d_frames.size() == window_size + 1 && state == CV_ONLY) {
            GyroBiasEstimation();
            SolveBAImu();
            state = TIGHTLY;
            SolveBA();
        }

        if(state == TIGHTLY) {
            SlidingWindow();
            Publish();
            //DrawUI();
        }
    }
}

bool BackEnd::GetNewKeyFrameAndMapPoints(FramePtr& keyframe, Eigen::VecVector3d& v_x3Dc) {
    if(state == TIGHTLY && marginalization_flag == MARGIN_OLD) {
        keyframe = new_keyframe;
        v_x3Dc = v_new_keyframe_x3Dc;
        return true;
    }
    return false;
}

BackEnd::MarginType BackEnd::AddFeaturesCheckParallax(FramePtr frame) {
    int last_track_num;
    int size_frames = d_frames.size();
    AddFeatures(frame, last_track_num);

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

void BackEnd::AddFeatures(FramePtr frame, int& last_track_num) {
    last_track_num = 0;
    int frame_idx = d_frames.size() - 1;
    for(int i = 0; i < frame->pt_id.size(); ++i) {
        auto it = m_features.find(frame->pt_id[i]);
        if(it == m_features.end()) {
            auto result = m_features.emplace(std::make_pair(frame->pt_id[i], Feature(frame->pt_id[i], frame_idx)));
            if(!result.second)
                throw std::runtime_error("m_features insert fail?");
            Feature& feat = result.first->second;
            feat.pt_n_per_frame.emplace_back(frame->pt_normal_plane[i]);
            feat.pt_r_n_per_frame.emplace_back(frame->pt_r_normal_plane[i]);
            feat.velocity_per_frame.emplace_back(0, 0, 0);
            feat.velocity_r_per_frame.emplace_back(0, 0, 0);
            feat.last_time = frame->timestamp;
            feat.last_pt_n = frame->pt_normal_plane[i];
            if(frame->pt_r_normal_plane[i](2) != 0) {
                feat.last_r_time = frame->timestamp;
                feat.last_pt_r_n = frame->pt_r_normal_plane[i];
            }
            else {
                feat.last_r_time = -1.0f;
            }
        }
        else {
            ++last_track_num;
            Feature& feat = it->second;
            feat.pt_n_per_frame.emplace_back(frame->pt_normal_plane[i]);
            feat.pt_r_n_per_frame.emplace_back(frame->pt_r_normal_plane[i]);
            double delta_t = frame->timestamp - feat.last_time;
            Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
            velocity.head<2>() = (frame->pt_normal_plane[i].head<2>() - feat.last_pt_n.head<2>()) / delta_t;
            feat.velocity_per_frame.emplace_back(velocity);
            feat.last_time = frame->timestamp;
            feat.last_pt_n = frame->pt_normal_plane[i];
            if(frame->pt_r_normal_plane[i](2) != 0) {
                if(feat.last_r_time == -1.0f) {
                    feat.velocity_r_per_frame.emplace_back(0, 0, 0);
                }
                else {
                    delta_t = frame->timestamp - feat.last_r_time;
                    velocity.head<2>() = (frame->pt_r_normal_plane[i].head<2>() - feat.last_pt_r_n.head<2>()) / delta_t;
                    feat.velocity_r_per_frame.emplace_back(velocity);
                }
                feat.last_r_time = frame->timestamp;
                feat.last_pt_r_n = frame->pt_r_normal_plane[i];
            }
            else {
                feat.velocity_r_per_frame.emplace_back(0, 0, 0);
            }
        }
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
                feat.velocity_per_frame.pop_front();
                feat.velocity_r_per_frame.pop_front();

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

    FramePtr frame_j = d_frames[d_frames.size() - 2];
    v_new_keyframe_x3Dc.clear();
    for(int i = 0, n = frame_j->pt_id.size(); i < n; ++i) {
        uint64_t pt_id = frame_j->pt_id[i];
        auto it = m_features.find(pt_id);
        if(it == m_features.end()) {
            v_new_keyframe_x3Dc.emplace_back(-1, -1, -1);
        }
        else {
            auto& feat = it->second;
            if(feat.inv_depth == -1.0) {
                v_new_keyframe_x3Dc.emplace_back(-1, -1, -1);
            }
            else {
                FramePtr frame_i = d_frames[feat.start_id];
                Eigen::Vector3d x3Dci = feat.pt_n_per_frame[0] / feat.inv_depth;
                Eigen::Vector3d x3Dbi = q_bc * x3Dci + p_bc;
                Eigen::Vector3d x3Dw = frame_i->q_wb * x3Dbi + frame_i->p_wb;
                Eigen::Vector3d x3Dbj = frame_j->q_wb.inverse() * (x3Dw - frame_j->p_wb);
                Eigen::Vector3d x3Dcj = q_bc.inverse() * (x3Dbj - p_bc);

                if(x3Dcj(2) <= 0) {
                    v_new_keyframe_x3Dc.emplace_back(-1, -1, -1);
                }
                else {
                    v_new_keyframe_x3Dc.emplace_back(x3Dcj);
                }
            }
        }
    }
    new_keyframe = frame_j;

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
                feat.velocity_per_frame.erase(feat.velocity_per_frame.begin() + j);
                feat.velocity_r_per_frame.erase(feat.velocity_r_per_frame.begin() + j);
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
        second_new->imupreinte->push_back(dt, acc, gyr);
    }

    latest_new->imupreinte = second_new->imupreinte;

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

            if(feat.pt_r_n_per_frame[i](2) != 0) {
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

        if(X(2) < 0.1) {
            feat.inv_depth = 1.0 / triangulate_default_depth;
        }
        else {
            feat.inv_depth = 1.0 / X(2);
        }
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
    last_margin_info = nullptr;
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

    std::memcpy(para_ex_bc, q_bc.data(), sizeof(double) * 4);
    std::memcpy(para_ex_bc + 4, p_bc.data(), sizeof(double) * 3);
    std::memcpy(para_ex_sm, q_rl.data(), sizeof(double) * 4);
    std::memcpy(para_ex_sm + 4, p_rl.data(), sizeof(double) * 3);

    if(estimate_time_delay) {
        para_Td[0] = time_delay;
    }
}

void BackEnd::double2data() {
    Sophus::SO3d q_w0b0 = d_frames[0]->q_wb;
    Eigen::Vector3d p_w0b0 = d_frames[0]->p_wb;

    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        size_t idx_pose = i * 7;
        size_t idx_vb = i * 9;
        std::memcpy(d_frames[i]->q_wb.data(), para_pose + idx_pose, sizeof(double) * Sophus::SO3d::num_parameters);
        std::memcpy(d_frames[i]->p_wb.data(), para_pose + idx_pose + 4, sizeof(double) * 3);
        std::memcpy(d_frames[i]->v_wb.data(), para_speed_bias + idx_vb , sizeof(double) * 3);
        std::memcpy(d_frames[i]->ba.data(), para_speed_bias + idx_vb + 3 , sizeof(double) * 3);
        std::memcpy(d_frames[i]->bg.data(), para_speed_bias + idx_vb + 6 , sizeof(double) * 3);
    }

    Sophus::SO3d q_w1b0 = d_frames[0]->q_wb;
    double y_diff = Sophus::R2ypr(q_w0b0 * q_w1b0.inverse())(0);
    Sophus::SO3d q_w0w1 = Sophus::ypr2R<double>(y_diff, 0, 0);
    Eigen::Vector3d p_w1b0 = d_frames[0]->p_wb;
    for(int i = 0, n = d_frames.size(); i < n; ++i) {
        d_frames[i]->q_wb = q_w0w1 * d_frames[i]->q_wb;
        d_frames[i]->p_wb = q_w0w1 * (d_frames[i]->p_wb - p_w1b0) + p_w0b0;
        d_frames[i]->v_wb = q_w0w1 * d_frames[i]->v_wb;
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
            v_outlier_feat_id.emplace_back(it.first);
        }
        else {
            // remove the point with bigger reprojection error
            int idx_i = feat.start_id;
            Eigen::Vector3d x3Dci = feat.pt_n_per_frame[0] / feat.inv_depth;
            Eigen::Vector3d x3Dbi = q_bc * x3Dci + p_bc;
            Eigen::Vector3d x3Dw = d_frames[idx_i]->q_wb * x3Dbi + d_frames[idx_i]->p_wb;
            double ave_reproj_error_norm = 0.0f;
            int num_factors = 0;

            for(int i = 0, n = feat.pt_n_per_frame.size(); i < n; ++i) {
                int idx_j = idx_i + i;
                Eigen::Vector3d x3Dbj = d_frames[idx_j]->q_wb.inverse() * (x3Dw - d_frames[idx_j]->p_wb);
                Eigen::Vector3d x3Dcj = q_bc.inverse() * (x3Dbj - p_bc);
                if(i != 0) {
                    Eigen::Vector2d residual = (x3Dcj.head<2>() / x3Dcj(2)) - feat.pt_n_per_frame[i].head<2>();
                    ave_reproj_error_norm += residual.norm();
                    ++num_factors;
                }

                if(feat.pt_r_n_per_frame[i](2) != 0) {
                    Eigen::Vector3d x3Drj = q_rl * x3Dcj + p_rl;
                    Eigen::Vector2d residual = (x3Drj.head<2>() / x3Drj(2)) - feat.pt_r_n_per_frame[i].head<2>();
                    ave_reproj_error_norm += residual.norm();
                    ++num_factors;
                }
            }

            ave_reproj_error_norm /= num_factors;
            if(ave_reproj_error_norm * focal_length > 3) {
                v_outlier_feat_id.emplace_back(it.first);
                //                LOG(INFO) << feat.feat_id << " " << ave_reproj_error_norm * focal_length;
            }
        }
    }

    for(auto& it : v_outlier_feat_id) {
        m_features.erase(it);
    }

    std::memcpy(q_bc.data(), para_ex_bc, sizeof(double) * 4);
    std::memcpy(p_bc.data(), para_ex_bc + 4, sizeof(double) * 3);
    std::memcpy(q_rl.data(), para_ex_sm, sizeof(double) * 4);
    std::memcpy(p_rl.data(), para_ex_sm + 4, sizeof(double) * 3);

    if(estimate_time_delay) {
        time_delay = para_Td[0];
    }
}

void BackEnd::SolveBA() {
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
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

            if(feat.pt_r_n_per_frame[j](2) != 0) {
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
    options.max_num_iterations = max_num_iterations;
    options.num_threads = 1;
    options.max_solver_time_in_seconds = max_solver_time_in_seconds; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    double2data();
}

void BackEnd::SolveBAImu() {
    Tracer::TraceBegin("BA");
    data2double();
    ceres::Problem problem;
    ceres::LossFunction *loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
    ceres::LossFunction *loss_function_lidar = new ceres::HuberLoss(0.1);
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


    //Lidar point info.

if(use_lidar_tracking){
    for(int i = 1, n = d_frames.size(); i < n; i = i + 4) {
        if(d_frames[i]->lidarInfo != -1 && d_frames[i-1]->lidarInfo != -1 ){
            //std::cout << "size ==" << d_frames[i]->cornerPointsLessSharp->points.size() << "\i=" << i <<std::endl;
            if(d_frames[i]->curr_point_edge.size() == 0){
                d_frames[i]->curr_point_edge.clear();
                d_frames[i]->last_point_a_edge.clear();
                d_frames[i]->last_point_b_edge.clear();
                FindEdgeNear3DPoint(d_frames[i-1], d_frames[i]);
            }
            ////printf("TransLidarPointToImage %f \n", t_opt.toc());
            for(int sharp = 0; sharp < d_frames[i]->curr_point_edge.size(); ++sharp){
                auto factor = new LidarEdgeFactor(d_frames[i]->curr_point_edge[sharp],
                                                  d_frames[i]->last_point_a_edge[sharp],
                                                  d_frames[i]->last_point_b_edge[sharp],
                                                  Tlb, 1, 0);
                problem.AddResidualBlock(factor, loss_function_lidar, para_pose + (i - 1) * 7, para_pose + i * 7);
            }

            if(d_frames[i]->curr_point_plane.size() == 0){
                d_frames[i]->curr_point_plane.clear();
                d_frames[i]->last_point_a_plane.clear();
                d_frames[i]->last_point_b_plane.clear();
                d_frames[i]->last_point_c_plane.clear();
                FindPlaneNear3DPoint(d_frames[i-1], d_frames[i]);
            }
            for(int plane = 0; plane < d_frames[i]->curr_point_plane.size(); ++plane){
                //std::cout << "curr_point_plane.size()=" << curr_point_plane.size() << std::endl;
                auto factor = new LidarPlaneFactor(d_frames[i]->curr_point_plane[plane],
                                                   d_frames[i]->last_point_a_plane[plane],
                                                   d_frames[i]->last_point_b_plane[plane],
                                                   d_frames[i]->last_point_c_plane[plane], Tlb, 1, 0);
                problem.AddResidualBlock(factor, loss_function_lidar, para_pose + (i - 1) * 7, para_pose + i * 7);
            }
        }
    }
}

    //int cornerPointsSharpNum = cornerPointsSharp->points.size();
    //int surfPointsFlatNum = surfPointsFlat->points.size();
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = max_num_iterations;
    options.num_threads = 4;
    //options.max_solver_time_in_seconds = max_solver_time_in_seconds; // 50 ms for solver and 50 ms for other
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    //std::cout << summary.FullReport() <<std::endl;
    double2data();

    for(int i = 1, n = d_frames.size(); i < n; ++i)
        d_frames[i]->imupreinte->repropagate(d_frames[i-1]->ba, d_frames[i-1]->bg);
    Tracer::TraceEnd();
    Marginalize();
}

void BackEnd::TransLidarPointToImage(FramePtr curframe, std::pair<sensor_msgs::PointCloud2ConstPtr, sensor_msgs::PointCloud2ConstPtr> lidarInfo){
    //TicToc t_opt;
    if(curframe->lidarInfo != -1){
        curframe->cornerPointsLessSharp = boost::make_shared<pcl::PointCloud<PointType>>();
        curframe->surfPointsLessFlat = boost::make_shared<pcl::PointCloud<PointType>>();
        curframe->kdtreeCornerLast = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZI>>();
        curframe->kdtreeSurfLast = boost::make_shared<pcl::KdTreeFLANN<pcl::PointXYZI>>();

        pcl::fromROSMsg(*lidarInfo.first, *curframe->cornerPointsLessSharp);
        pcl::fromROSMsg(*lidarInfo.second, *curframe->surfPointsLessFlat);
        int cornerPointsSharpNum = curframe->cornerPointsLessSharp->points.size();
        int surfPointsFlatNum = curframe->surfPointsLessFlat->points.size();
        Sophus::SE3d Twb_img(curframe->q_wb, curframe->p_wb);
        Sophus::SE3d Twb_lidar(curframe->q_wb_lidar, curframe->p_wb_lidar);
        Sophus::SE3d Ti_l = Tlb * Twb_img.inverse() * Twb_lidar * Tlb.inverse();
        //convert Xlidar to image timestamp.
        for(int i = 0; i<cornerPointsSharpNum; i++){
            Eigen::Vector3d point(curframe->cornerPointsLessSharp->points[i].x,
                                  curframe->cornerPointsLessSharp->points[i].y,
                                  curframe->cornerPointsLessSharp->points[i].z);
            Eigen::Vector3d pointInimage = Ti_l * point;
            curframe->cornerPointsLessSharp->points[i].x = pointInimage.x();
            curframe->cornerPointsLessSharp->points[i].y = pointInimage.y();
            curframe->cornerPointsLessSharp->points[i].z = pointInimage.z();
        }

        for(int i = 0; i<surfPointsFlatNum; i++){
            Eigen::Vector3d point(curframe->surfPointsLessFlat->points[i].x,
                                  curframe->surfPointsLessFlat->points[i].y,
                                  curframe->surfPointsLessFlat->points[i].z);
            Eigen::Vector3d pointInimage = Ti_l * point ;
            curframe->surfPointsLessFlat->points[i].x = pointInimage.x();
            curframe->surfPointsLessFlat->points[i].y = pointInimage.y();
            curframe->surfPointsLessFlat->points[i].z = pointInimage.z();
        }

        curframe->kdtreeCornerLast->setInputCloud(curframe->cornerPointsLessSharp);
        curframe->kdtreeSurfLast->setInputCloud(curframe->surfPointsLessFlat );
    }
    //printf("TransLidarPointToImage %f \n", t_opt.toc());
}


void BackEnd::FindPlaneNear3DPoint(FramePtr last_frame, FramePtr frame){
    int surfPointsFlatNum = frame->surfPointsLessFlat->points.size();
    pcl::PointXYZI pointSel;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    Sophus::SE3d Twb_imgj(frame->q_wb, frame->p_wb);
    Sophus::SE3d Twb_imgi(last_frame->q_wb, last_frame->p_wb);
    Sophus::SE3d Ti_l = Tlb * Twb_imgi.inverse() * Twb_imgj * Tlb.inverse();
    int plane_correspondence = 0;
    //if(surfPointsFlatNum > 1000)
    //    surfPointsFlatNum = 1000;
    for (int i = 0; i < surfPointsFlatNum; i++)
    {
        TransformToStart(&(frame->surfPointsLessFlat->points[i]), &pointSel, Ti_l);
        last_frame->kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
        if (pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD)
        {
            closestPointInd = pointSearchInd[0];

            // get closest point's scan ID
            int closestPointScanID = int(last_frame->surfPointsLessFlat->points[closestPointInd].intensity);
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD, minPointSqDis3 = DISTANCE_SQ_THRESHOLD;

            // search in the direction of increasing scan line
            for (int j = closestPointInd + 1; j < (int)last_frame->surfPointsLessFlat->points.size(); ++j)
            {
                // if not in nearby scans, end the loop
                if (int(last_frame->surfPointsLessFlat->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                    break;

                double pointSqDis = (last_frame->surfPointsLessFlat->points[j].x - pointSel.x) *
                        (last_frame->surfPointsLessFlat->points[j].x - pointSel.x) +
                        (last_frame->surfPointsLessFlat->points[j].y - pointSel.y) *
                        (last_frame->surfPointsLessFlat->points[j].y - pointSel.y) +
                        (last_frame->surfPointsLessFlat->points[j].z - pointSel.z) *
                        (last_frame->surfPointsLessFlat->points[j].z - pointSel.z);

                // if in the same or lower scan line
                if (int(last_frame->surfPointsLessFlat->points[j].intensity) <= closestPointScanID && pointSqDis < minPointSqDis2)
                {
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
                // if in the higher scan line
                else if (int(last_frame->surfPointsLessFlat->points[j].intensity) > closestPointScanID && pointSqDis < minPointSqDis3)
                {
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }

            // search in the direction of decreasing scan line
            for (int j = closestPointInd - 1; j >= 0; --j)
            {
                // if not in nearby scans, end the loop
                if (int(last_frame->surfPointsLessFlat->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                    break;

                double pointSqDis = (last_frame->surfPointsLessFlat->points[j].x - pointSel.x) *
                        (last_frame->surfPointsLessFlat->points[j].x - pointSel.x) +
                        (last_frame->surfPointsLessFlat->points[j].y - pointSel.y) *
                        (last_frame->surfPointsLessFlat->points[j].y - pointSel.y) +
                        (last_frame->surfPointsLessFlat->points[j].z - pointSel.z) *
                        (last_frame->surfPointsLessFlat->points[j].z - pointSel.z);

                // if in the same or higher scan line
                if (int(last_frame->surfPointsLessFlat->points[j].intensity) >= closestPointScanID && pointSqDis < minPointSqDis2)
                {
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
                else if (int(last_frame->surfPointsLessFlat->points[j].intensity) < closestPointScanID && pointSqDis < minPointSqDis3)
                {
                    // find nearer point
                    minPointSqDis3 = pointSqDis;
                    minPointInd3 = j;
                }
            }
            if (minPointInd2 >= 0 && minPointInd3 >= 0)
            {

                Eigen::Vector3d curr_point_tmp(frame->surfPointsLessFlat->points[i].x,
                                               frame->surfPointsLessFlat->points[i].y,
                                               frame->surfPointsLessFlat->points[i].z);
                Eigen::Vector3d last_point_a_tmp(last_frame->surfPointsLessFlat->points[closestPointInd].x,
                                                 last_frame->surfPointsLessFlat->points[closestPointInd].y,
                                                 last_frame->surfPointsLessFlat->points[closestPointInd].z);
                Eigen::Vector3d last_point_b_tmp(last_frame->surfPointsLessFlat->points[minPointInd2].x,
                                                 last_frame->surfPointsLessFlat->points[minPointInd2].y,
                                                 last_frame->surfPointsLessFlat->points[minPointInd2].z);
                Eigen::Vector3d last_point_c_tmp(last_frame->surfPointsLessFlat->points[minPointInd3].x,
                                                 last_frame->surfPointsLessFlat->points[minPointInd3].y,
                                                 last_frame->surfPointsLessFlat->points[minPointInd3].z);
                frame->curr_point_plane.emplace_back(curr_point_tmp);
                frame->last_point_a_plane.emplace_back(last_point_a_tmp);
                frame->last_point_b_plane.emplace_back(last_point_b_tmp);
                frame->last_point_c_plane.emplace_back(last_point_c_tmp);
                //s_plane.emplace_back(1);
                plane_correspondence++;

            }

        }
    }
    //std::cout << "plane_correspondence=" << plane_correspondence <<std::endl;
    //std::cout << "plane_all=" << surfPointsFlatNum <<std::endl;
}


void BackEnd::FindEdgeNear3DPoint(FramePtr last_frame, FramePtr frame){

    int cornerPointsSharpNum = frame->cornerPointsLessSharp->points.size();
    pcl::PointXYZI pointSel;
    std::vector<int> pointSearchInd;
    std::vector<float> pointSearchSqDis;
    Sophus::SE3d Twb_imgj(frame->q_wb, frame->p_wb);
    Sophus::SE3d Twb_imgi(last_frame->q_wb, last_frame->p_wb);
    Sophus::SE3d Ti_l = Tlb * Twb_imgi.inverse() * Twb_imgj * Tlb.inverse();
    int edge_correspondence = 0;
    //if(cornerPointsSharpNum > 600)
    //    cornerPointsSharpNum = 600;
    for (int i = 0; i < cornerPointsSharpNum; i++)
    {
        TransformToStart(&(frame->cornerPointsLessSharp->points[i]), &pointSel, Ti_l);
        last_frame->kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);
        int closestPointInd = -1, minPointInd2 = -1;
        if(pointSearchSqDis[0] < DISTANCE_SQ_THRESHOLD) {
            closestPointInd = pointSearchInd[0];
            int closestPointScanID = int(last_frame->cornerPointsLessSharp->points[closestPointInd].intensity);
            double minPointSqDis2 = DISTANCE_SQ_THRESHOLD;
            //search increse
            for (int j = closestPointInd + 1; j < (int)last_frame->cornerPointsLessSharp->points.size(); ++j)
            {
                // if in the same scan line, continue
                if (int(last_frame->cornerPointsLessSharp->points[j].intensity) <= closestPointScanID)
                    continue;

                // if not in nearby scans, end the loop
                if (int(last_frame->cornerPointsLessSharp->points[j].intensity) > (closestPointScanID + NEARBY_SCAN))
                    break;

                double pointSqDis = (last_frame->cornerPointsLessSharp->points[j].x - pointSel.x) *
                        (last_frame->cornerPointsLessSharp->points[j].x - pointSel.x) +
                        (last_frame->cornerPointsLessSharp->points[j].y - pointSel.y) *
                        (last_frame->cornerPointsLessSharp->points[j].y - pointSel.y) +
                        (last_frame->cornerPointsLessSharp->points[j].z - pointSel.z) *
                        (last_frame->cornerPointsLessSharp->points[j].z - pointSel.z);

                if (pointSqDis < minPointSqDis2)
                {
                    // find nearer point
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            }
            //search decreas
            for (int j = closestPointInd - 1; j >= 0; --j)
            {
                // if in the same scan line, continue
                if (int(last_frame->cornerPointsLessSharp->points[j].intensity) >= closestPointScanID)
                    continue;

                // if not in nearby scans, end the loop
                if (int(last_frame->cornerPointsLessSharp->points[j].intensity) < (closestPointScanID - NEARBY_SCAN))
                    break;

                double pointSqDis = (last_frame->cornerPointsLessSharp->points[j].x - pointSel.x) *
                        (last_frame->cornerPointsLessSharp->points[j].x - pointSel.x) +
                        (last_frame->cornerPointsLessSharp->points[j].y - pointSel.y) *
                        (last_frame->cornerPointsLessSharp->points[j].y - pointSel.y) +
                        (last_frame->cornerPointsLessSharp->points[j].z - pointSel.z) *
                        (last_frame->cornerPointsLessSharp->points[j].z - pointSel.z);

                if (pointSqDis < minPointSqDis2)
                {
                    // find nearer point
                    minPointSqDis2 = pointSqDis;
                    minPointInd2 = j;
                }
            }
        }
        //step2. if find the mid and closest point.
        //insert to cost function, 1.cur_point, 2.cloest_point 3. mid_point 4.realtime(s)
        if (minPointInd2 >= 0) // both closestPointInd and minPointInd2 is valid
        {
            Eigen::Vector3d curr_point(frame->cornerPointsLessSharp->points[i].x,
                                       frame->cornerPointsLessSharp->points[i].y,
                                       frame->cornerPointsLessSharp->points[i].z);
            Eigen::Vector3d last_point_a_tmp(last_frame->cornerPointsLessSharp->points[closestPointInd].x,
                                             last_frame->cornerPointsLessSharp->points[closestPointInd].y,
                                             last_frame->cornerPointsLessSharp->points[closestPointInd].z);
            Eigen::Vector3d last_point_b_tmp(last_frame->cornerPointsLessSharp->points[minPointInd2].x,
                                             last_frame->cornerPointsLessSharp->points[minPointInd2].y,
                                             last_frame->cornerPointsLessSharp->points[minPointInd2].z);

            frame->curr_point_edge.push_back(curr_point);
            frame->last_point_a_edge.push_back(last_point_a_tmp);
            frame->last_point_b_edge.push_back(last_point_b_tmp);
            //s_edge.emplace_back(1);
            edge_correspondence++;
        }
    }
    //std::cout << "edge_correspondence=" << edge_correspondence <<std::endl;
    //std::cout << "edge_all=" << cornerPointsSharpNum <<std::endl;
}

void BackEnd::TransformToStart(PointType const *const pi, PointType *const po, Sophus::SE3d &lidari_T_lidarj){
    //interpolation ratio
    double s;
    if (0)
        s = (pi->intensity - int(pi->intensity)) / SCAN_PERIOD;
    else
        s = 1.0;
    //s = 1;
    Eigen::Quaterniond q_last_curr  = lidari_T_lidarj.unit_quaternion();
    Eigen::Quaterniond q_point_last = Eigen::Quaterniond::Identity().slerp(s, q_last_curr);
    Eigen::Vector3d t_point_last = s * lidari_T_lidarj.translation();
    Eigen::Vector3d point(pi->x, pi->y, pi->z);
    Eigen::Vector3d un_point = q_point_last * point + t_point_last;

    po->x = un_point.x();
    po->y = un_point.y();
    po->z = un_point.z();
    po->intensity = pi->intensity;
}

void BackEnd::SolvePnP(FramePtr frame) {
    // pnp this help debug
    std::vector<cv::Point2f> v_pts;
    std::vector<cv::Point3d> v_mps;

    for(int i = 0, n = frame->pt_id.size(); i < n; ++i) {
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
        IntegrationBasePtr imu_preintegration = frame_i1->imupreinte;
        Eigen::Matrix3d JR_bg = imu_preintegration->jacobian.block<3, 3>(O_R, O_BG);
        Sophus::SO3d delRi_i1 = imu_preintegration->delta_q;
        Sophus::SO3d qwi = frame_i->q_wb, qwi1 = frame_i1->q_wb;
        Eigen::Vector3d residual = (delRi_i1.inverse() * (qwi1.inverse() * qwi)).log();
        err_0 += residual.squaredNorm();
        H += JR_bg.transpose() * JR_bg;
        b += JR_bg.transpose() * residual;
    }

    Eigen::Vector3d bg = d_frames[0]->bg + H.ldlt().solve(b);

    for(auto& frame : d_frames) {
        frame->imupreinte->repropagate(Eigen::Vector3d::Zero(), bg);
        frame->bg = bg;
    }

    for(int i = 0, n = d_frames.size(); i < n - 1; ++i) {
        FramePtr frame_i = d_frames[i];
        FramePtr frame_i1 = d_frames[i + 1];
        Sophus::SO3d delRi_i1 = frame_i1->imupreinte->delta_q;
        Sophus::SO3d qwi = frame_i->q_wb, qwi1 = frame_i1->q_wb;
        Eigen::Vector3d residual = (delRi_i1.inverse() * (qwi1.inverse() * qwi)).log();
        err_1 += residual.squaredNorm();
    }

    LOG(INFO) << "bg: " << bg(0) << " " << bg(1) << " " << bg(2) << ", err: " << err_0 << " -> " << err_1;
    return true;
}

Sophus::SO3d BackEnd::InitFirstIMUPose(const Eigen::VecVector3d& v_acc) {
    Sophus::SO3d qwb;
    Eigen::Vector3d ave_acc = Eigen::Vector3d::Zero();
    int n = v_acc.size();
    for(int i = 0; i < n; ++i) {
        ave_acc += v_acc[i];
    }
    ave_acc /= n;

    Eigen::Vector3d ngb = ave_acc.normalized();
    Eigen::Vector3d ngw(0, 0, 1.0f);
    Eigen::Vector3d ypr = Sophus::R2ypr(Eigen::Quaterniond::FromTwoVectors(ngb, ngw));
    qwb = Sophus::ypr2R<double>(0, ypr(1), ypr(2));
    return qwb;
}



void BackEnd::PredictNextLidarPose(FramePtr cur_frame, Eigen::VecVector3d& lidar_acc, Eigen::VecVector3d& lidar_gyro, std::vector<double>& lidar_imut){
    double t0 = cur_frame->v_imu_timestamp.back();

    Eigen::Vector3d gyr_0 = cur_frame->v_gyr.back(), acc_0 = cur_frame->v_acc.back();
    Sophus::SO3d  q_wb_tmp = cur_frame->q_wb;
    Eigen::Vector3d p_wb_tmp = cur_frame->p_wb, v_wb_tmp = cur_frame->v_wb;
    double close_t = 0.02;
    //Predict pose and implement imu preintegration.
    for(int i = 0, n = lidar_imut.size(); i < n; ++i) {
        double t = lidar_imut[i], dt = t - t0;

        Eigen::Vector3d gyr = lidar_gyro[i];
        Eigen::Vector3d acc = lidar_acc[i];
        Eigen::Vector3d un_acc_0 = q_wb_tmp * (acc_0 - cur_frame->ba) - gw;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr) - cur_frame->bg;
        q_wb_tmp = q_wb_tmp * Sophus::SO3d::exp(un_gyr * dt);
        Eigen::Vector3d un_acc_1 = q_wb_tmp * (acc - cur_frame->ba) - gw;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        p_wb_tmp += dt * v_wb_tmp + 0.5 * dt * dt * un_acc;
        v_wb_tmp += dt * un_acc;
        if(fabs(cur_frame->lidar_time - t) < close_t){
            close_t = fabs(cur_frame->lidar_time - t);
            cur_frame->p_wb_lidar = p_wb_tmp;
            cur_frame->q_wb_lidar = q_wb_tmp;
            cur_frame->lidarInfo = 2;
        }
        gyr_0 = gyr;
        acc_0 = acc;
        t0 = t;
    }
}

void BackEnd::PredictNextFramePose(FramePtr ref_frame, FramePtr cur_frame) {
    cur_frame->q_wb = ref_frame->q_wb;
    cur_frame->p_wb = ref_frame->p_wb;
    cur_frame->v_wb = ref_frame->v_wb;
    cur_frame->ba = ref_frame->ba;
    cur_frame->bg = ref_frame->bg;
    cur_frame->imupreinte = std::make_shared<IntegrationBase>(
                ref_frame->v_acc.back(), ref_frame->v_gyr.back(),
                ref_frame->ba, ref_frame->bg,
                acc_n, gyr_n, acc_w, gyr_w);

    Eigen::Vector3d gyr_0 = ref_frame->v_gyr.back(), acc_0 = ref_frame->v_acc.back();
    double t0 = ref_frame->v_imu_timestamp.back();


    //set imu preintegration info for lidar pose predict.
    {
        imuinfo_tmp.header.frame_id = "world";
        imuinfo_tmp.header.stamp.fromSec(t0);
        imuinfo_tmp.BackTime = t0;

        //wvio_T_bodyCur
        imuinfo_tmp.pose.orientation.w = ref_frame->q_wb.unit_quaternion().w();
        imuinfo_tmp.pose.orientation.x = ref_frame->q_wb.unit_quaternion().x();
        imuinfo_tmp.pose.orientation.y = ref_frame->q_wb.unit_quaternion().y();
        imuinfo_tmp.pose.orientation.z = ref_frame->q_wb.unit_quaternion().z();
        imuinfo_tmp.pose.position.x = ref_frame->p_wb.x();
        imuinfo_tmp.pose.position.y = ref_frame->p_wb.y();
        imuinfo_tmp.pose.position.z = ref_frame->p_wb.z();

        imuinfo_tmp.bias_acc.x = ref_frame->ba.x();
        imuinfo_tmp.bias_acc.y = ref_frame->ba.y();
        imuinfo_tmp.bias_acc.z = ref_frame->ba.z();

        imuinfo_tmp.bias_gyro.x = ref_frame->bg.x();
        imuinfo_tmp.bias_gyro.y = ref_frame->bg.y();
        imuinfo_tmp.bias_gyro.z = ref_frame->bg.z();

        imuinfo_tmp.velocity.x = ref_frame->v_wb.x();
        imuinfo_tmp.velocity.y = ref_frame->v_wb.y();
        imuinfo_tmp.velocity.z = ref_frame->v_wb.z();

        imuinfo_tmp.acc_0.x = ref_frame->v_acc.back().x();
        imuinfo_tmp.acc_0.y = ref_frame->v_acc.back().y();
        imuinfo_tmp.acc_0.z = ref_frame->v_acc.back().z();

        imuinfo_tmp.gyr_0.x = ref_frame->v_gyr.back().x();
        imuinfo_tmp.gyr_0.y = ref_frame->v_gyr.back().y();
        imuinfo_tmp.gyr_0.z = ref_frame->v_gyr.back().z();
    }



    //Predict pose and implement imu preintegration.
    double close_t = 0.02;
    for(int i = 0, n = cur_frame->v_imu_timestamp.size(); i < n; ++i) {
        double t = cur_frame->v_imu_timestamp[i], dt = t - t0;
        Eigen::Vector3d gyr = cur_frame->v_gyr[i];
        Eigen::Vector3d acc = cur_frame->v_acc[i];
        cur_frame->imupreinte->push_back(dt, acc, gyr);
        Eigen::Vector3d un_acc_0 = cur_frame->q_wb * (acc_0 - ref_frame->ba) - gw;
        Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr) - ref_frame->bg;
        cur_frame->q_wb = cur_frame->q_wb * Sophus::SO3d::exp(un_gyr * dt);
        Eigen::Vector3d un_acc_1 = cur_frame->q_wb * (acc - ref_frame->ba) - gw;
        Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        cur_frame->p_wb += dt * cur_frame->v_wb + 0.5 * dt * dt * un_acc;
        cur_frame->v_wb += dt * un_acc;
        if(fabs(cur_frame->lidar_time - t) < close_t && cur_frame->lidar_time != -1){
            close_t = fabs(cur_frame->lidar_time - t);
            Sophus::SO3d q_tmp = cur_frame->q_wb;
            Eigen::Vector3d p_tmp = cur_frame->p_wb;
            cur_frame->p_wb_lidar = p_tmp;
            cur_frame->q_wb_lidar = q_tmp;
            cur_frame->lidarInfo = 1;
        }
        gyr_0 = gyr;
        acc_0 = acc;
        t0 = t;
    }
}

void BackEnd::Marginalize() {
    if(marginalization_flag == MARGIN_OLD) {
        ScopedTrace st("margin_old");
        margin_mps.clear();
        auto loss_function = new ceres::HuberLoss(cv_huber_loss_parameter);
        auto loss_function_lidar = new ceres::HuberLoss(0.1);
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

        //lidar
if(use_lidar_tracking){
        if(d_frames[1]->curr_point_edge.size() > 200 && d_frames[1]->curr_point_plane.size() > 200){

            for(int sharp = 0; sharp < d_frames[1]->curr_point_edge.size(); sharp++ ){
                auto factor = new LidarEdgeFactor(d_frames[1]->curr_point_edge[sharp],
                        d_frames[1]->last_point_a_edge[sharp],
                        d_frames[1]->last_point_b_edge[sharp],
                        Tlb, 1, 0);
                auto residual_block_info = new ResidualBlockInfo(factor, loss_function_lidar,
                                                                 std::vector<double*>{para_pose,
                                                                                      para_pose + 7},
                                                                 std::vector<int>{0});
                margin_info->addResidualBlockInfo(residual_block_info);
            }

            for(int plane = 0; plane < d_frames[1]->curr_point_plane.size(); plane++ ){
                auto factor = new LidarPlaneFactor(d_frames[1]->curr_point_plane[plane],
                        d_frames[1]->last_point_a_plane[plane],
                        d_frames[1]->last_point_b_plane[plane],
                        d_frames[1]->last_point_c_plane[plane],
                        Tlb, 1, 0);
                auto residual_block_info = new ResidualBlockInfo(factor, loss_function_lidar,
                                                                 std::vector<double*>{para_pose,
                                                                                      para_pose + 7}, std::vector<int>{0});
                margin_info->addResidualBlockInfo(residual_block_info);
            }

        }
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

void BackEnd::Publish() {
    if(!pub_vio_Twc.empty()) {
        FramePtr frame = d_frames.back();
        Sophus::SE3d Twc = Sophus::SE3d(frame->q_wb, frame->p_wb) * Sophus::SE3d(q_bc, p_bc);
        for(auto& pub : pub_vio_Twc)
            pub(frame->timestamp, Twc);

        for(auto& pub : pub_vio_Twb)
            pub(frame->timestamp, Sophus::SE3d(frame->q_wb, frame->p_wb));
    }

    //pub imu preintegration info for vloam
    if(imuinfo_tmp.BackTime > 0) {
        ImuPreintInfo(imuinfo_tmp.BackTime, imuinfo_tmp);
    }

    for(auto& pub : pub_frame) {
        pub(d_frames.back());
    }

    if(!pub_mappoints.empty()) {
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

        for(auto& pub : pub_mappoints) {
            pub(mps);
        }
    }
}

//void BackEnd::DrawUI() {
//    if(d_frames.empty())
//        return;

//    auto frame = d_frames.back();

//    if(draw_pose) {
//        draw_pose(frame->id, frame->timestamp, Sophus::SE3d(frame->q_wb * q_bc, frame->q_wb * p_bc + frame->p_wb));
//    }

//    if(draw_mps) {
//        Eigen::VecVector3d mps;

//        for(auto& it : m_features) {
//            auto& feat = it.second;
//            if(feat.inv_depth == -1.0f)
//                continue;
//            Sophus::SE3d Twb(d_frames[feat.start_id]->q_wb, d_frames[feat.start_id]->p_wb);
//            Sophus::SE3d Tbc(q_bc, p_bc);
//            Eigen::Vector3d x3Dc = feat.pt_n_per_frame[0] / feat.inv_depth;
//            Eigen::Vector3d x3Dw = Twb * Tbc * x3Dc;
//            mps.emplace_back(x3Dw);
//        }

//        draw_mps(frame->id, frame->timestamp, mps);
//    }

//    if(draw_sw) {
//        std::vector<Sophus::SE3d> v_Twc;

//        for(int i = 0, n = d_frames.size(); i < n - 1; ++i) {
//            v_Twc.emplace_back(d_frames[i]->q_wb * q_bc, d_frames[i]->q_wb * p_bc + d_frames[i]->p_wb);
//        }

//        draw_sw(frame->id, frame->timestamp, v_Twc);
//    }

//    if(draw_margin_mps && (marginalization_flag == MARGIN_OLD)) {
//        draw_margin_mps(frame->id, frame->timestamp, margin_mps);
//    }
//}
