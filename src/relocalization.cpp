#include "relocalization.h"
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>
#include <ceres/ceres.h>
#include "ceres/pose_graph.h"

Relocalization::Relocalization(const std::string& voc_filename, const std::string& brief_pattern_file,
                               CameraPtr camera_, const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_)
    : camera(camera_), q_bc(q_bc_), p_bc(p_bc_)
{
    voc = std::make_shared<BriefVocabulary>(voc_filename);
    db.setVocabulary(*voc, false, 0);
    pose_graph_thread = std::thread(&Relocalization::Optimize4DoF, this);

    // load brief pattern
    cv::FileStorage fs(brief_pattern_file, cv::FileStorage::READ);
    if(!fs.isOpened()) {
        throw std::runtime_error("Could not open the BRIEF pattern file: " + brief_pattern_file);
    }

    std::vector<int> x1, y1, x2, y2;
    fs["x1"] >> x1;
    fs["x2"] >> x2;
    fs["y1"] >> y1;
    fs["y2"] >> y2;

    brief_extractor[0].importPairs(x1, y1, x2, y2);
    brief_extractor[1].importPairs(x1, y1, x2, y2);
}

void Relocalization::UpdateVIOPose(double timestamp, const Sophus::SE3d& T_viow_c) {
    mtx_w_viow.lock();
    Sophus::SE3d Tw_viow(q_w_viow, p_w_viow);
    mtx_w_viow.unlock();
    Sophus::SE3d Twc = Tw_viow * T_viow_c;

    for(auto& pub : pub_reloc_Twc) {
        pub(timestamp, Twc);
    }
}

Sophus::SE3d Relocalization::ShiftPoseWorld(const Sophus::SE3d& Tviow) {
    mtx_w_viow.lock();
    Sophus::SE3d Tw_viow(q_w_viow, p_w_viow);
    mtx_w_viow.unlock();
    return Tw_viow * Tviow;
}

Eigen::Vector3d Relocalization::ShiftVectorWorld(const Eigen::Vector3d& Vviow) {
    std::unique_lock<std::mutex> lock(mtx_w_viow);
    return q_w_viow * Vviow;
}

void Relocalization::ProcessFrame(FramePtr frame) {
    mtx_w_viow.lock();
    Sophus::SE3d Tw_viow(q_w_viow, p_w_viow);
    mtx_w_viow.unlock();
    Sophus::SE3d Twc = Tw_viow * Sophus::SE3d(frame->vio_q_wb, frame->vio_p_wb) * Sophus::SE3d(q_bc, p_bc);
    mtx_reloc_path.lock();
    for(auto& pub : pub_add_reloc_path)
        pub(Twc);
    mtx_reloc_path.unlock();

    int64_t candidate_index = DetectLoop(frame);
    if(candidate_index == -1)
        return;
    FramePtr old_frame = v_frame_database[candidate_index];

    if(FindMatchesAndSolvePnP(old_frame, frame)) {
        for(auto& pub : pub_loop_edge)
            pub(std::make_pair(old_frame->frame_id, frame->frame_id));
        mtx_optimize_buffer.lock();
        v_optimize_buffer.emplace_back(frame->frame_id);
        mtx_optimize_buffer.unlock();
        cv_optimize_buffer.notify_one();
    }
}

int64_t Relocalization::DetectLoop(FramePtr frame) {
    ScopedTrace st("detect_loop");
    // re-id
    frame->frame_id = next_frame_id++;

    // extract extra fast corner
    std::vector<cv::KeyPoint> v_keypoint, v_extra_keypoint;
    cv::FAST(frame->img, v_extra_keypoint, 20, true);
    // extract feature descriptor
    for(int i = 0, n = frame->v_pt_2d_uv.size(); i < n; ++i) {
        cv::KeyPoint kp;
        kp.pt = frame->v_pt_2d_uv[i];
        v_keypoint.emplace_back(kp);
    }

    brief_extractor[0].compute(frame->img, v_keypoint, frame->v_descriptor);
    brief_extractor[1].compute(frame->img, v_extra_keypoint, frame->v_extra_descriptor);

    for(int i = 0, n = v_extra_keypoint.size(); i < n; ++i) {
        cv::Point2f pt = v_extra_keypoint[i].pt;
        frame->v_extra_pt_2d_uv.emplace_back(pt);
        Eigen::Vector3d pt_normal;
        camera->BackProject(Eigen::Vector2d(pt.x, pt.y), pt_normal);
        frame->v_extra_pt_2d_normal.emplace_back(pt_normal(0), pt_normal(1));
    }
    // end of extract feature

    // release img memory
    // for debug using resize img
    frame->img = cv::Mat();

    DBoW2::QueryResults ret;
    db.query(frame->v_extra_descriptor, ret, 4, frame->frame_id - 50);
    db.add(frame->v_extra_descriptor);

    mtx_frame_database.lock();
    v_frame_database.emplace_back(frame);
    mtx_frame_database.unlock();

    if(frame->frame_id <= 50)
        return -1;

    // detect loop step 1, check bag of word similar
    // a good match with its neighbor
    // 0.05 is magic number
    // 0.015 is also magic number
    bool find_loop = false;
    int min_index = std::numeric_limits<int>::max();
    if(ret.size() > 1 && ret[0].Score > 0.05)
        for(int i = 1, n = ret.size(); i < n; ++i) {
            if(ret[i].Score > 0.015) {
                find_loop = true;
                if(ret[i].Id < min_index) {
                    min_index = ret[i].Id;
                }
            }
        }

    if(find_loop)
        return min_index;
    else
        return -1;
}

bool Relocalization::FindMatchesAndSolvePnP(FramePtr old_frame, FramePtr frame) {
    ScopedTrace st("match_pnp");
    std::vector<cv::Point2f> matched_2d_cur, matched_2d_old;
    std::vector<cv::Point2f> matched_2d_cur_norm, matched_2d_old_norm;
    std::vector<cv::Point3f> matched_3d;
    std::vector<uint64_t> matched_id;
    std::vector<uchar> status;

    for(int i = 0, n = frame->v_descriptor.size(); i < n; ++i) {
        int best_dis = 80;
        int best_index = -1;
        for(int j = 0, m = old_frame->v_extra_descriptor.size(); j < m; ++j) {
            int dis = DVision::BRIEF::distance(frame->v_descriptor[i], old_frame->v_extra_descriptor[j]);
            if(dis < best_dis) {
                best_dis = dis;
                best_index = j;
            }
        }

        if(best_index != -1) {
            matched_id.emplace_back(frame->v_pt_id[i]);
            matched_2d_cur.emplace_back(frame->v_pt_2d_uv[i]);
            matched_2d_cur_norm.emplace_back(frame->v_pt_2d_normal[i]);
            matched_3d.emplace_back(frame->v_pt_3d[i]);
            matched_2d_old.emplace_back(old_frame->v_extra_pt_2d_uv[best_index]);
            matched_2d_old_norm.emplace_back(old_frame->v_extra_pt_2d_normal[best_index]);
        }
    }

    if(matched_2d_cur.size() > 25) { // 25 is magic number
        Sophus::SO3d q_w_c0 = old_frame->vio_q_wb * q_bc;
        Sophus::SO3d q_w_c1 = frame->vio_q_wb * q_bc;
        Eigen::Vector3d p_w_c0 = old_frame->vio_q_wb * p_bc + old_frame->vio_p_wb;
        Eigen::Vector3d p_w_c1 = frame->vio_q_wb * p_bc + frame->vio_p_wb;
        Sophus::SO3d q_c0_c1 = q_w_c0.inverse() * q_w_c1;
        Eigen::Vector3d p_c0_c1 = q_w_c0.inverse() * (p_w_c1 - p_w_c0);

        cv::Mat R, rvec, tvec;
        cv::eigen2cv(q_c0_c1.matrix(), R);
        cv::Rodrigues(R, rvec);
        cv::eigen2cv(p_c0_c1, tvec);
        cv::Mat inliers;
        if(cv::solvePnPRansac(matched_3d, matched_2d_old_norm, cv::Mat::eye(3, 3, CV_64F), cv::noArray(),
                              rvec, tvec, false, 100, 8.0f / camera->f(), 0.99, inliers)) {

            status.resize(matched_2d_cur.size(), 0);

            for(int i = 0, n = inliers.rows; i < n; ++i)
                status[inliers.at<int>(i)] = 1;

            util::ReduceVector(matched_2d_cur, status);
            util::ReduceVector(matched_2d_old, status);

            if(matched_2d_cur.size() >= 25) {

                Eigen::Vector3d tmp_t;
                Eigen::Matrix3d tmp_R;
                cv::Rodrigues(rvec, R);
                cv::cv2eigen(R, tmp_R);
                cv::cv2eigen(tvec, tmp_t);
                Sophus::SE3d Tco_cc(Eigen::Quaterniond(tmp_R), tmp_t); // camera_old, camera_cur
                Sophus::SE3d Tbc(q_bc, p_bc);
                Sophus::SE3d Tbo_bc = Tbc * Tco_cc * Tbc.inverse(); // body_old, body_cur
                frame->pnp_p_old_cur = Tbo_bc.translation();
                frame->pnp_q_old_cur = Tbo_bc.so3();

                // compute relative_yaw
                double yaw_w_old = Sophus::R2ypr(old_frame->vio_q_wb * frame->pnp_q_old_cur.inverse())(0);
                double yaw_w_cur = Sophus::R2ypr(frame->vio_q_wb)(0);
                double yaw = NormalizeAngle(yaw_w_cur - yaw_w_old); // old_cur
                frame->pnp_yaw_old_cur = yaw;

                if(std::abs(yaw) < M_PI / 6 && tmp_t.norm() < 20.0) {
                    cv::Mat result;
                    cv::hconcat(old_frame->compressed_img, frame->compressed_img, result);

                    for(int i = 0, n = matched_2d_old.size(); i < n; ++i) {
                        cv::Point2f pt_i = matched_2d_old[i] / 2;
                        cv::Point2f pt_j = matched_2d_cur[i] / 2;
                        int width = result.cols / 2;
                        pt_j.x += width;
                        cv::circle(result, pt_i, 2, cv::Scalar(0, 255, 255), -1);
                        cv::circle(result, pt_j, 2, cv::Scalar(0, 255, 255), -1);
                        cv::line(result, pt_i, pt_j, cv::Scalar(0, 255, 0), 1);
                    }

                    frame->has_loop = true;
                    frame->loop_index = old_frame->frame_id;
                    frame->matched_img = result;

                    for(auto& pub : pub_reloc_img) {
                        pub(frame->matched_img);
                    }

                    return true;
                }
            }
        }
    }
    return false;
}

void Relocalization::Optimize4DoF() {
    while(1) {
        int64_t cur_index = -1;
        std::unique_lock<std::mutex> lock(mtx_optimize_buffer);
        cv_optimize_buffer.wait(lock, [&] {
            if(!v_optimize_buffer.empty()) {
                cur_index = v_optimize_buffer.back();
                v_optimize_buffer.clear();
            }
            return cur_index != -1;
        });
        lock.unlock();
        Tracer::TraceBegin("pose_graph");
        LOG(INFO) << "optimize pose graph";
        std::unique_lock<std::mutex> lock1(mtx_frame_database);

        int size = cur_index + 1;
        // thank c++11 =)
        double p_wb_raw[size * 3];
        double ypr_wb_raw[size * 3];

        ceres::Problem problem;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization* angle_local_parameterization = AngleLocalParameterization::Create();

        bool first_loop_detect = false;
        for(int i = 0; i <= cur_index; ++i) {
            Eigen::Vector3d p_w_bj = v_frame_database[i]->vio_p_wb;
            Sophus::SO3d q_w_bj = v_frame_database[i]->vio_q_wb;
            p_wb_raw[i * 3] = p_w_bj(0);
            p_wb_raw[i * 3 + 1] = p_w_bj(1);
            p_wb_raw[i * 3 + 2] = p_w_bj(2);

            Eigen::Vector3d ypr_w_bj = Sophus::R2ypr(q_w_bj);
            ypr_wb_raw[i * 3] = ypr_w_bj(0);
            ypr_wb_raw[i * 3 + 1] = ypr_w_bj(1);
            ypr_wb_raw[i * 3 + 2] = ypr_w_bj(2);

            problem.AddParameterBlock(ypr_wb_raw + 3 * i, 1, angle_local_parameterization);
            problem.AddParameterBlock(p_wb_raw + 3 * i, 3);

            // add edge
            for(int j = 1; j < 5; ++j) {
                if(i - j < 0)
                    continue;
                Eigen::Vector3d p_w_bi = v_frame_database[i - j]->vio_p_wb;
                Sophus::SO3d q_w_bi = v_frame_database[i - j]->vio_q_wb;
                Eigen::Vector3d p_bi_bj = q_w_bi.inverse() * (p_w_bj - p_w_bi);
                Sophus::SO3d q_bi_bj = q_w_bi.inverse() * q_w_bj;
                Eigen::Vector3d ypr_w_bi = Sophus::R2ypr(q_w_bi);
                double yaw_bi_bj = NormalizeAngle(ypr_w_bj(0) - ypr_w_bi(0));
                auto factor = FourDOFError::Create(p_bi_bj(0), p_bi_bj(1), p_bi_bj(2),
                                                   yaw_bi_bj, ypr_w_bi(1), ypr_w_bi(2));
                problem.AddResidualBlock(factor, NULL,
                                         ypr_wb_raw + 3 * (i - j), p_wb_raw + 3 * (i - j),
                                         ypr_wb_raw + 3 * i, p_wb_raw + 3 * i);
            }

            // add loop
            if(v_frame_database[i]->has_loop) {
                int64_t loop_index = v_frame_database[i]->loop_index;

                if(!first_loop_detect) {
                    first_loop_detect = true;
                    problem.SetParameterBlockConstant(ypr_wb_raw + 3 * loop_index);
                    problem.SetParameterBlockConstant(p_wb_raw + 3 * loop_index);
                }

                Sophus::SO3d q_w_bi = v_frame_database[loop_index]->vio_q_wb;
                Eigen::Vector3d ypr_w_bi = Sophus::R2ypr(q_w_bi);
                double pnp_yaw_bi_bj = v_frame_database[i]->pnp_yaw_old_cur;
                Eigen::Vector3d pnp_p_bi_bj = v_frame_database[i]->pnp_p_old_cur;
                auto factor = FourDOFError::Create(pnp_p_bi_bj(0), pnp_p_bi_bj(1), pnp_p_bi_bj(2),
                                                   pnp_yaw_bi_bj, ypr_w_bi(1), ypr_w_bi(2),
                                                   10.0, 15.0);
                problem.AddResidualBlock(factor, loss_function,
                                         ypr_wb_raw + 3 * loop_index,
                                         p_wb_raw + 3 * loop_index,
                                         ypr_wb_raw + 3 * i,
                                         p_wb_raw + 3 * i);
            }
        }

        lock1.unlock();
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 5;
        options.num_threads = 1;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
//        LOG(INFO) << summary.FullReport();

        lock1.lock();
        for(int i = 0; i <= cur_index; ++i) {
            v_frame_database[i]->p_wb(0) = p_wb_raw[i * 3];
            v_frame_database[i]->p_wb(1) = p_wb_raw[i * 3 + 1];
            v_frame_database[i]->p_wb(2) = p_wb_raw[i * 3 + 2];
            v_frame_database[i]->q_wb = Sophus::ypr2R(ypr_wb_raw[i * 3],
                                                      ypr_wb_raw[i * 3 + 1],
                                                      ypr_wb_raw[i * 3 + 2]);
        }

        // update the world and vio world transformation matrix
        {
            double relative_yaw = Sophus::R2ypr(v_frame_database[cur_index]->q_wb)(0) - Sophus::R2ypr(v_frame_database[cur_index]->vio_q_wb)(0);
            mtx_w_viow.lock();
            q_w_viow = Sophus::ypr2R<double>(relative_yaw, 0, 0);
            p_w_viow = v_frame_database[cur_index]->p_wb - q_w_viow * v_frame_database[cur_index]->vio_p_wb;
            mtx_w_viow.unlock();

            for(int i = cur_index + 1, n = v_frame_database.size(); i < n; ++i) {
                Eigen::Vector3d p_viow_b = v_frame_database[i]->vio_p_wb;
                Sophus::SO3d q_viow_b = v_frame_database[i]->vio_q_wb;

                v_frame_database[i]->p_wb = q_w_viow * p_viow_b + p_w_viow;
                v_frame_database[i]->q_wb = q_w_viow * q_viow_b;
            }
        }

        if(!pub_update_reloc_path.empty()) {
            mtx_reloc_path.lock();
            std::vector<Sophus::SE3d> v_Twc;
            for(auto& frame : v_frame_database) {
                Sophus::SE3d Twc = Sophus::SE3d(frame->q_wb, frame->p_wb) * Sophus::SE3d(q_bc, p_bc);
                v_Twc.emplace_back(Twc);
            }
            mtx_reloc_path.unlock();

            for(auto& pub : pub_update_reloc_path)
                pub(v_Twc);
        }
        lock1.unlock();

        Tracer::TraceEnd();
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }
}
