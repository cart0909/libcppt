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
    detect_loop_thread = std::thread(&Relocalization::Process, this);
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

void Relocalization::PushFrame(FramePtr frame) {
    m_frame_buffer.lock();
    v_frame_buffer.emplace_back(frame);
    m_frame_buffer.unlock();
    cv_frame_buffer.notify_one();
}

void Relocalization::Process() {
    while(1) {
        std::vector<FramePtr> v_frames;
        std::unique_lock<std::mutex> lock(m_frame_buffer);
        cv_frame_buffer.wait(lock, [&] {
           v_frames = v_frame_buffer;
           v_frame_buffer.clear();
           return !v_frames.empty();
        });
        lock.unlock();

        for(auto& frame : v_frames) {
            ProcessFrame(frame);
        }
    }
}

void Relocalization::ProcessFrame(FramePtr frame) {
    std::unique_lock<std::mutex> lock(mtx_frame_database);
    int64_t candidate_index = DetectLoop(frame);
    if(candidate_index == -1)
        return;
    FramePtr old_frame = v_frame_database[candidate_index];

    if(FindMatchesAndSolvePnP(old_frame, frame)) {
        lock.unlock();
        // assume old_frame(0) vio_p_wb and vio_q_wb
        // world is true world
        //        Eigen::Vector3d p_w_b0 = old_frame->vio_p_wb;
        //        Sophus::SO3d q_w_b0 = old_frame->vio_q_wb;

        //        // and cur_frame(1) vio_p_wb and vio_q_wb
        //        // world is drift world, d is drift world
        //        Eigen::Vector3d p_d_b1 = frame->vio_p_wb;
        //        Sophus::SO3d q_d_b1 = frame->vio_q_wb;

        //        Eigen::Vector3d p_w_b1 = q_w_b0 * frame->pnp_p_old_cur + p_w_b0;
        //        Sophus::SO3d q_w_b1 = q_w_b0 * frame->pnp_q_old_cur;

        //        // then we calc the true world and drift world transformation matrix
        //        Eigen::Vector3d p_b1_d = -(q_d_b1.inverse() * p_d_b1);
        //        Eigen::Vector3d p_wd = q_w_b1 * (p_b1_d) + p_w_b1;
        //        Sophus::SO3d q_wd = q_w_b1 * q_d_b1.inverse();

        //        // strong assume!!!
        //        // world and drift world only exist x, y, z, yaw angle drift
        //        {
        //            double yaw_drift = Sophus::R2ypr(q_wd)(0);
        //            if(yaw_drift * 2 > M_PI)
        //                yaw_drift -= M_PI;
        //            q_wd = Sophus::ypr2R<double>(yaw_drift, 0, 0);
        //        }

        // shift all vio pose of whole sequence to the world frame
        //        for(auto& it : v_frame_database) {
        //            Eigen::Vector3d p_db = it->vio_p_wb;
        //            Sophus::SO3d q_db = it->vio_q_wb;
        //            it->vio_q_wb = q_wd * q_db;
        //            it->vio_p_wb = q_wd * p_db + p_wd;
        //        }
        mtx_optimize_buffer.lock();
        v_optimize_buffer.emplace_back(frame->frame_id);
        mtx_optimize_buffer.unlock();
        cv_optimize_buffer.notify_one();
    }
}

int64_t Relocalization::DetectLoop(FramePtr frame) {
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
    cv::resize(frame->img, frame->compressed_img, frame->img.size() / 2);
    cv::cvtColor(frame->compressed_img, frame->compressed_img, CV_GRAY2BGR);
    frame->img = cv::Mat();


    DBoW2::QueryResults ret;
    db.query(frame->v_extra_descriptor, ret, 4, frame->frame_id - 50);

    db.add(frame->v_extra_descriptor);
    v_frame_database.emplace_back(frame);

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
        Sophus::SO3d q_w_c0 = old_frame->q_wb * q_bc;
        Sophus::SO3d q_w_c1 = frame->q_wb * q_bc;
        Eigen::Vector3d p_w_c0 = old_frame->q_wb * p_bc + old_frame->p_wb;
        Eigen::Vector3d p_w_c1 = frame->q_wb * p_bc + frame->p_wb;
        Sophus::SO3d q_c0_c1 = q_w_c0.inverse() * q_w_c1;
        Eigen::Vector3d p_c0_c1 = q_w_c0.inverse() * (p_w_c1 - p_w_c0);

        cv::Mat R, rvec, tvec;
        cv::eigen2cv(q_c0_c1.matrix(), R);
        cv::Rodrigues(R, rvec);
        cv::eigen2cv(p_c0_c1, tvec);
        cv::Mat inliers;
        if(cv::solvePnPRansac(matched_3d, matched_2d_old_norm, cv::Mat::eye(3, 3, CV_64F), cv::noArray(),
                              rvec, tvec, true, 100, 8.0f / camera->f(), 0.99, inliers)) {

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
                frame->pnp_p_old_cur = tmp_t;
                frame->pnp_q_old_cur.setQuaternion(Eigen::Quaterniond(tmp_R));
                double yaw = Sophus::R2ypr(frame->pnp_q_old_cur)(0);
                if(yaw * 2 > M_PI)
                    yaw -= M_PI;
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
                    frame->matched_img = result;

//                    Eigen::Vector3d ypr = Sophus::R2ypr(frame->pnp_q_old_cur) * 180 / M_PI;
//                    LOG(INFO) << ypr(0) << " " << ypr(1) << " " << ypr(2) << " " <<
//                                 tmp_t(0) << " " << tmp_t(1) << " " << tmp_t(2);
                    return true;
                }
            }
        }
    }
    return false;
}

void Relocalization::Optimize4DoF() {
    while(1) {
        bool optimize_flag = false;
        uint64_t cur_index = 0;
        std::unique_lock<std::mutex> lock(mtx_optimize_buffer);
        cv_optimize_buffer.wait(lock, [&] {
            if(!v_optimize_buffer.empty()) {
                optimize_flag = true;
                cur_index = v_optimize_buffer.back();
                v_optimize_buffer.clear();
            }
            return optimize_flag;
        });
        lock.unlock();

        LOG(INFO) << "optimize pose graph";
        std::unique_lock<std::mutex> lock1(mtx_frame_database);

        int size = cur_index + 1;
        double* p_wb_raw = new double[size * 3];
        double* yaw_wb_raw = new double[size];

        ceres::Problem problem;
        ceres::LossFunction *loss_function = new ceres::HuberLoss(0.1);
        ceres::LocalParameterization* angle_local_parameterization = AngleLocalParameterization::Create();

        for(int i = 0; i < cur_index; ++i) {
            Eigen::Vector3d p_wb = v_frame_database[i]->vio_p_wb;
            Sophus::SO3d q_wb = v_frame_database[i]->vio_q_wb;
            p_wb_raw[i * 3] = p_wb(0);
            p_wb_raw[i * 3 + 1] = p_wb(1);
            p_wb_raw[i * 3 + 2] = p_wb(2);

            yaw_wb_raw[i] = Sophus::R2ypr(q_wb)(0);
            if(yaw_wb_raw[i] * 2 > M_PI)
                yaw_wb_raw[i] -= M_PI;

            problem.AddParameterBlock(yaw_wb_raw + i, 1, angle_local_parameterization);
            problem.AddParameterBlock(p_wb_raw + 3 * i, 3);

            // add edge

            // add loop
        }

        lock1.unlock();
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
        options.max_num_iterations = 5;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        LOG(INFO) << summary.FullReport();

        lock1.lock();

        lock1.unlock();
    }
}
