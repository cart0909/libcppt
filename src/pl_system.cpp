#include "pl_system.h"
#include "converter.h"
#include <opencv2/core/eigen.hpp>
#include <opencv2/ccalib/omnidir.hpp>
#include <future>

PLSystem::PLSystem() {}

PLSystem::PLSystem(const std::string& config_file) {
    cv::setNumThreads(0);
    reset_flag = false;
    param = ConfigLoader::Load(config_file);
    InitCameraParameters();

    feature_tracker = std::make_shared<FeatureTracker>(cam_m, param.clahe_parameter, param.fast_threshold,
                                                       param.min_dist, param.Fundamental_reproj_threshold);
    line_tracker = std::make_shared<LineTracker>(cam_m);

    if(param.estimate_extrinsic)
        stereo_matcher = std::make_shared<StereoMatcher>(cam_m, cam_s, param.clahe_parameter, param.Fundamental_reproj_threshold);
    else
        stereo_matcher = std::make_shared<StereoMatcher>(cam_m, cam_s, param.p_rl[0], param.q_rl[0],
                                                         param.clahe_parameter, param.Fundamental_reproj_threshold);

    line_stereo_matcher = std::make_shared<LineStereoMatcher>(cam_m, cam_s);

    backend = std::make_shared<PLBackEnd>(cam_m->f(), param.gyr_noise, param.acc_noise,
                                          param.gyr_bias_noise, param.acc_bias_noise,
                                          param.p_rl[0], param.p_bc[0], param.q_rl[0], param.q_bc[0],
                                          param.gravity_magnitude, param.sliding_window_size, param.keyframe_parallax,
                                          param.max_solver_time_in_seconds, param.max_num_iterations, param.cv_huber_loss_parameter,
                                          param.triangulate_default_depth, param.max_imu_sum_t, param.min_init_stereo_num, param.estimate_extrinsic,
                                          param.estimate_td, param.init_td);

    if(param.enable_reloc)
        reloc = std::make_shared<Relocalization>(param.voc_filename, param.brief_pattern_file, cam_m,
                                                 param.q_bc[0], param.p_bc[0]);
    if(reloc) {
        backend->SubVIOTwc(std::bind(&Relocalization::UpdateVIOPose, reloc,
                                     std::placeholders::_1,
                                     std::placeholders::_2));
        reloc_thread = std::thread(&PLSystem::RelocProcess, this);
    }

    frontend_thread = std::thread(&PLSystem::FrontEndProcess, this);
    backend_thread = std::thread(&PLSystem::BackEndProcess, this);
    backend_busy = false;
}

PLSystem::~PLSystem() {}

void PLSystem::FrontEndProcess() {
    while(1) {
        cv::Mat img_l, img_r;
        double timestamp = -1.0f;
        std::unique_lock<std::mutex> lock(mtx_frontend);
        cv_frontend.wait(lock, [&] {
            if(!frontend_buffer_img.empty()) {
                img_l = std::get<0>(frontend_buffer_img.back());
                img_r = std::get<1>(frontend_buffer_img.back());
                timestamp = std::get<2>(frontend_buffer_img.back());
                frontend_buffer_img.clear();
            }
            return (timestamp != -1.0f);
        });
        lock.unlock();

        if(reset_flag) {
            LOG(WARNING) << "reset the system...";
            reset_flag = false;
            b_first_frame = true;
            m_id_history.clear();
            m_id_optical_flow.clear();
            backend->ResetRequest();
        }

        Tracer::TraceBegin("remap");
        cv::remap(img_l, img_l, M1l, M2l, CV_INTER_LINEAR);
        Tracer::TraceEnd();
        Tracer::TraceBegin("remap");
        cv::remap(img_r, img_r, M1r, M2r, CV_INTER_LINEAR);
        Tracer::TraceEnd();

        FeatureTracker::FramePtr feat_frame;
        LineTracker::FramePtr line_frame;
        if(b_first_frame) {
            feat_frame = feature_tracker->InitFirstFrame(img_l, timestamp);
            line_frame = line_tracker->InitFirstFrame(img_l, timestamp);
            b_first_frame = false;
        }
        else {
            feat_frame = feature_tracker->Process(img_l, timestamp);
            line_frame = line_tracker->Process(img_l, timestamp);
        }

        if(!backend_busy) {
            mtx_backend.lock();
            backend_buffer_img.emplace_back(feat_frame, line_frame, img_r);
            mtx_backend.unlock();
            cv_backend.notify_one();
        }
    }
}

void PLSystem::BackEndProcess() {
    while(1) {
        Eigen::VecVector3d v_gyr, v_acc;
        std::vector<double> v_imu_t;
        FeatureTracker::FramePtr feat_frame;
        LineTracker::FramePtr line_frame;
        cv::Mat img_r;
        double img_t = -1.0f;
        double td = backend->GetTd();
        std::unique_lock<std::mutex> lock(mtx_backend);
        cv_backend.wait(lock, [&] {
            while(1) {
                if(backend_buffer_img.empty() || backend_buffer_imu_t.empty())
                    return false;

                if(backend_buffer_imu_t.back() < std::get<0>(backend_buffer_img.front())->timestamp + td)
                    return false;

                if(backend_buffer_imu_t.front() > std::get<0>(backend_buffer_img.front())->timestamp + td) {
                    backend_buffer_img.pop_front();
                    continue;
                }

                feat_frame = std::get<0>(backend_buffer_img.front());
                line_frame = std::get<1>(backend_buffer_img.front());
                img_r = std::get<2>(backend_buffer_img.front());
                img_t = feat_frame->timestamp;
                backend_buffer_img.pop_front();

                while(backend_buffer_imu_t.front() < img_t + td) {
                    v_gyr.emplace_back(backend_buffer_gyr.front());
                    v_acc.emplace_back(backend_buffer_acc.front());
                    v_imu_t.emplace_back(backend_buffer_imu_t.front());
                    backend_buffer_gyr.pop_front();
                    backend_buffer_acc.pop_front();
                    backend_buffer_imu_t.pop_front();
                }

                return true;
            }
        });
        lock.unlock();
        backend_busy = true;

        feature_tracker->ExtractFAST(feat_frame);
        PubTrackingImg(feat_frame, line_frame);
        StereoMatcher::FramePtr stereo_frame = stereo_matcher->Process(feat_frame, img_r);
        LineStereoMatcher::FramePtr line_stereo_frame = line_stereo_matcher->Process(line_frame, img_r);
        PLBackEnd::FramePtr back_frame = Converter::Convert(feat_frame, stereo_frame, line_frame,
                                                            line_stereo_frame, cam_m, cam_s,
                                                            v_gyr, v_acc, v_imu_t);

        back_frame->td = td;
        backend->ProcessFrame(back_frame);

        if(reloc) {
            BackEnd::FramePtr new_keyframe;
            Eigen::VecVector3d v_x3Dc;
            if(backend->GetNewKeyFrameAndMapPoints(new_keyframe, v_x3Dc)) {
                mtx_reloc.lock();
                reloc_buffer_keyframe.emplace_back(new_keyframe, v_x3Dc);
                mtx_reloc.unlock();
                cv_reloc.notify_one();
            }

            mtx_reloc.lock();
            reloc_buffer_frame.emplace_back(feat_frame, back_frame);
            mtx_reloc.unlock();
            cv_reloc.notify_one();
        }
        backend_busy = false;
    }
}

void PLSystem::PubTrackingImg(FeatureTracker::FramePtr feat_frame, LineTracker::FramePtr line_frame) {
    if(!pub_tracking_img.empty()) {
        std::map<uint64_t, cv::Point2f> m_id_history_tmp;
        std::map<uint64_t, std::shared_ptr<std::deque<cv::Point2f>>> m_id_optical_flow_tmp;
        for(int i = 0, n = feat_frame->pt_id.size(); i < n; ++i) {
            auto it = m_id_history.find(feat_frame->pt_id[i]);
            if(it != m_id_history.end()) {
                m_id_history_tmp[it->first] = it->second;
                auto d_pt = m_id_optical_flow[it->first];
                m_id_optical_flow_tmp[it->first] = d_pt;
                d_pt->emplace_back(feat_frame->pt[i]);
                while(d_pt->size() > 2) {
                    d_pt->pop_front();
                }
            }
            else {
                m_id_history_tmp[feat_frame->pt_id[i]] = feat_frame->pt[i];
                m_id_optical_flow_tmp[feat_frame->pt_id[i]] = std::make_shared<std::deque<cv::Point2f>>();
                m_id_optical_flow_tmp[feat_frame->pt_id[i]]->emplace_back(feat_frame->pt[i]);
            }
        }
        m_id_history = std::move(m_id_history_tmp);
        m_id_optical_flow = std::move(m_id_optical_flow_tmp);

        cv::Mat tracking_img = feat_frame->compressed_img.clone();
        for(auto& it : m_id_optical_flow) {
            uint64_t pt_id = it.first;
            std::deque<cv::Point2f>& history_pts = *it.second;
            for(int i = 0, n = history_pts.size() - 1; i < n; ++i) {
                cv::line(tracking_img, history_pts[i] / 2, history_pts[i+1] / 2, cv::Scalar(0, 255, 255), 2);
            }
            cv::circle(tracking_img, history_pts.back() / 2, 2, cv::Scalar(0, 255, 0), -1);
        }

        for(auto& line : line_frame->v_lines) {
            cv::Point2f sp(line.startPointX, line.startPointY);
            cv::Point2f ep(line.endPointX, line.endPointY);
            cv::line(tracking_img, sp / 2, ep / 2, cv::Scalar(255, 102, 255), 2);
            cv::circle(tracking_img, sp / 2, 2, cv::Scalar(0, 0, 255), -1);
            cv::circle(tracking_img, ep / 2, 2, cv::Scalar(255, 0, 0), -1);
        }

        for(auto& pub : pub_tracking_img) {
            pub(feat_frame->timestamp, tracking_img);
        }
    }
}

void PLSystem::InitCameraParameters() {
    cv::Mat K1, K2, D1, D2, R1, R2, P1, P2;
    cv::Mat R, t;
    K1 = (cv::Mat_<double>(3, 3) << param.intrinsic_master[0][0], 0, param.intrinsic_master[0][2],
                                    0, param.intrinsic_master[0][1], param.intrinsic_master[0][3],
                                    0,                            0,                            1);

    K2 = (cv::Mat_<double>(3, 3) << param.intrinsic_slave[0][0], 0, param.intrinsic_slave[0][2],
                                    0, param.intrinsic_slave[0][1], param.intrinsic_slave[0][3],
                                    0,                           0,                           1);

    D1 = (cv::Mat_<double>(4, 1) << param.distortion_master[0][0], param.distortion_master[0][1],
                                    param.distortion_master[0][2], param.distortion_master[0][3]);

    D2 = (cv::Mat_<double>(4, 1) << param.distortion_slave[0][0], param.distortion_slave[0][1],
                                    param.distortion_slave[0][2], param.distortion_slave[0][3]);

    cv::eigen2cv(param.q_rl[0].matrix(), R);
    cv::eigen2cv(param.p_rl[0], t);

    cv::Size image_size(param.width[0], param.height[0]);
    if(param.cam_model[0] == "PINHOLE") {
        cv::stereoRectify(K1, D1, K2, D2, image_size, R, t, R1, R2, P1, P2, cv::noArray());
        cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32F, M1l, M2l);
        cv::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_32F, M1r, M2r);


        cam_m = CameraPtr(new IdealPinhole(image_size.width, image_size.height,
                                           P1.at<double>(0, 0), P1.at<double>(1, 1),
                                           P1.at<double>(0, 2), P1.at<double>(1, 2)));
        cam_s = cam_m;
    }
    else if(param.cam_model[0] == "FISHEYE") {
        cv::fisheye::stereoRectify(K1, D1, K2, D2, image_size, R, t, R1, R2, P1, P2, cv::noArray(), cv::CALIB_ZERO_DISPARITY);
        double baseline = -P2.at<double>(0, 3)/P2.at<double>(0, 0);
        P1.at<double>(0, 0) = K1.at<double>(0, 0);
        P1.at<double>(1, 1) = K1.at<double>(1, 1);
        P2.at<double>(0, 0) = K1.at<double>(0, 0);
        P2.at<double>(1, 1) = K1.at<double>(1, 1);
        P2.at<double>(0, 3) = -baseline * P2.at<double>(0, 0);

        cv::fisheye::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32F, M1l, M2l);
        cv::fisheye::initUndistortRectifyMap(K2, D2, R2, P2, image_size, CV_32F, M1r, M2r);

        cam_m = CameraPtr(new IdealPinhole(image_size.width, image_size.height,
                                           P1.at<double>(0, 0), P1.at<double>(1, 1),
                                           P1.at<double>(0, 2), P1.at<double>(1, 2)));
        cam_s = cam_m;
    }
    else if(param.cam_model[0] == "OMNI") {
        // TODO
    }

    Sophus::SE3d T_lp_l, T_rp_r;
    {
        Eigen::Matrix3d R_lp_l, R_rp_r;
        cv::cv2eigen(R1, R_lp_l);
        cv::cv2eigen(R2, R_rp_r);
        T_lp_l.setRotationMatrix(R_lp_l);
        T_rp_r.setRotationMatrix(R_rp_r);
    }

    Sophus::SE3d T_bl(param.q_bc[0], param.p_bc[0]), T_rl(param.q_rl[0], param.p_rl[0]);
    Sophus::SE3d T_rp_lp = T_rp_r * T_rl * T_lp_l.inverse();
    Sophus::SE3d T_b_lp = T_bl * T_lp_l.inverse();
    param.q_bc[0] = T_b_lp.so3();
    param.p_bc[0] = T_b_lp.translation();
    param.q_rl[0] = T_rp_lp.so3();
    param.p_rl[0] = T_rp_lp.translation();
}
