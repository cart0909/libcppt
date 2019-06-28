#include "system.h"
#include "converter.h"
#include <functional>
#include <glog/logging.h>

System::System() {}

System::System(const std::string& config_file) {
    cv::setNumThreads(0);
    reset_flag = false;
    param = ConfigLoader::Load(config_file);
    InitCameraParameters();
    feature_tracker = std::make_shared<FeatureTracker>(cam_m, param.clahe_parameter, param.fast_threshold,
                                                       param.min_dist, param.Fundamental_reproj_threshold);

    if(param.estimate_extrinsic)
        stereo_matcher = std::make_shared<StereoMatcher>(cam_m, cam_s, param.clahe_parameter, param.Fundamental_reproj_threshold);
    else
        stereo_matcher = std::make_shared<StereoMatcher>(cam_m, cam_s, param.p_rl[0], param.q_rl[0],
                param.clahe_parameter, param.Fundamental_reproj_threshold);

    backend = std::make_shared<BackEnd>(cam_m->f(), param.gyr_noise, param.acc_noise,
                                        param.gyr_bias_noise, param.acc_bias_noise,
                                        param.p_rl[0], param.p_bc[0], param.q_rl[0], param.q_bc[0],
            param.gravity_magnitude, param.sliding_window_size, param.keyframe_parallax,
            param.max_solver_time_in_seconds, param.max_num_iterations, param.cv_huber_loss_parameter,
            param.triangulate_default_depth, param.max_imu_sum_t, param.min_init_stereo_num, param.estimate_extrinsic,
            param.estimate_td, param.init_td, param.use_lidar_tracking);
    use_lidar_tracking = param.use_lidar_tracking;

    if(param.enable_reloc)
        reloc = std::make_shared<Relocalization>(param.voc_filename, param.brief_pattern_file, cam_m,
                                                 param.q_bc[0], param.p_bc[0]);
    if(reloc) {
        backend->SubVIOTwc(std::bind(&Relocalization::UpdateVIOPose, reloc,
                                     std::placeholders::_1,
                                     std::placeholders::_2));
        backend->SubVIOTwb(std::bind(&Relocalization::UpdateVIOPoseWB, reloc,
                                     std::placeholders::_1,
                                     std::placeholders::_2));
        reloc_thread = std::thread(&System::RelocProcess, this);
    }

    frontend_thread = std::thread(&System::FrontEndProcess, this);
    backend_thread = std::thread(&System::BackEndProcess, this);
    backend_busy = false;
}

System::~System() {}

void System::InitCameraParameters() {
    if(param.cam_model[0] == "PINHOLE") {
        cam_m = CameraPtr(new Pinhole(param.width[0], param.height[0],
                param.intrinsic_master[0][0], param.intrinsic_master[0][1],
                param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                param.distortion_master[0][0], param.distortion_master[0][1],
                param.distortion_master[0][2], param.distortion_master[0][3]));

        cam_s = CameraPtr(new Pinhole(param.width[0], param.height[0],
                param.intrinsic_slave[0][0], param.intrinsic_slave[0][1],
                param.intrinsic_slave[0][2], param.intrinsic_slave[0][3],
                param.distortion_slave[0][0], param.distortion_slave[0][1],
                param.distortion_slave[0][2], param.distortion_slave[0][3]));
    }
    else if(param.cam_model[0] == "FISHEYE") {
        cam_m = CameraPtr(new Fisheye(param.width[0], param.height[0],
                param.intrinsic_master[0][0], param.intrinsic_master[0][1],
                param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                param.distortion_master[0][0], param.distortion_master[0][1],
                param.distortion_master[0][2], param.distortion_master[0][3]));

        cam_s = CameraPtr(new Fisheye(param.width[0], param.height[0],
                param.intrinsic_slave[0][0], param.intrinsic_slave[0][1],
                param.intrinsic_slave[0][2], param.intrinsic_slave[0][3],
                param.distortion_slave[0][0], param.distortion_slave[0][1],
                param.distortion_slave[0][2], param.distortion_slave[0][3]));
    }
    else if(param.cam_model[0] == "OMNI") {
        cam_m = CameraPtr(new Omni(param.width[0], param.height[0],
                param.intrinsic_master[0][0], param.intrinsic_master[0][1],
                param.intrinsic_master[0][2], param.intrinsic_master[0][3],
                param.intrinsic_master[0][4],
                param.distortion_master[0][0], param.distortion_master[0][1],
                param.distortion_master[0][2], param.distortion_master[0][3]));
        cam_s = CameraPtr(new Omni(param.width[0], param.height[0],
                param.intrinsic_slave[0][0], param.intrinsic_slave[0][1],
                param.intrinsic_slave[0][2], param.intrinsic_slave[0][3],
                param.intrinsic_slave[0][4],
                param.distortion_slave[0][0], param.distortion_slave[0][1],
                param.distortion_slave[0][2], param.distortion_slave[0][3]));
    }
}

void System::Reset() {
    reset_flag = true;
    LOG(WARNING) << "system reset flag is turn on!";
}

void System::PushImages(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp) {
    mtx_frontend.lock();
    frontend_buffer_img.emplace_back(img_l, img_r, timestamp);
    mtx_frontend.unlock();
    cv_frontend.notify_one();
}

void System::PushImuData(const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double timestamp) {
    mtx_backend.lock();
    backend_buffer_gyr.emplace_back(gyr);
    backend_buffer_acc.emplace_back(acc);
    backend_buffer_imu_t.emplace_back(timestamp);
    mtx_backend.unlock();
    cv_backend.notify_one();
}

void System::PushLidarData(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharp2,
                           const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlat2){
    mtx_backend.lock();
    backend_buffer_lidar.emplace_back(cornerPointsLessSharp2, surfPointsLessFlat2);
    mtx_backend.unlock();
    cv_backend.notify_one();
}


void System::FrontEndProcess() {
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

        FeatureTracker::FramePtr feat_frame;
        if(b_first_frame) {
            feat_frame = feature_tracker->InitFirstFrame(img_l, timestamp);
            b_first_frame = false;
        }
        else {
            feat_frame = feature_tracker->Process(img_l, timestamp);
        }

        if(!backend_busy) {
            mtx_backend.lock();
            backend_buffer_img.emplace_back(feat_frame, img_r);
            mtx_backend.unlock();
            cv_backend.notify_one();
        }
    }
}

void System::BackEndProcess() {
    while(1) {
        Eigen::VecVector3d v_gyr, v_acc;
        Eigen::VecVector3d v_gyr_lidar, v_acc_lidar;
        std::vector<double> v_imu_t_lidar;
        std::vector<double> v_imu_t;
        v_gyr_lidar.clear(); v_acc_lidar.clear();
        v_imu_t_lidar.clear();
        FeatureTracker::FramePtr feat_frame;
        std::pair<sensor_msgs::PointCloud2ConstPtr, sensor_msgs::PointCloud2ConstPtr> lidar_info;
        bool start_this_frame = false;
        cv::Mat img_r;
        double img_t = -1.0f;
        double td = backend->GetTd();
        std::unique_lock<std::mutex> lock(mtx_backend);
        cv_backend.wait(lock, [&] {
            while(1) {
                if(backend_buffer_img.empty() || backend_buffer_imu_t.empty() || backend_buffer_lidar.empty())
                    return false;

                if(backend_buffer_imu_t.back() < backend_buffer_img.front().first->timestamp + td)
                    return false;

                if(backend_buffer_imu_t.front() > backend_buffer_img.front().first->timestamp + td) {
                    backend_buffer_img.pop_front();
                    continue;
                }

                feat_frame = backend_buffer_img.front().first;
                img_r = backend_buffer_img.front().second;
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

                //std::cout << "img_t=" << img_t <<std::endl;

                //find close lidar info.
                if(!backend_buffer_lidar.empty() && use_lidar_tracking!=0){
                    int close_id = -1;
                    double close_time = 0.05;
                    for(int i = 0; i<backend_buffer_lidar.size(); i++){
                        if(fabs(backend_buffer_lidar[i].first->header.stamp.toSec() - (img_t + td)) < close_time){
                            close_time = fabs(backend_buffer_lidar[i].first->header.stamp.toSec() - (img_t + td));
                            close_id = i;
                            start_this_frame = true;
                        }
                    }
                    //int ind= lower_bound(backend_buffer_lidar.begin() ,backend_buffer_lidar.end(),
                    //                     img_t + td, comp);
                    //std::cout << "ind=" << ind <<std::endl;
                    //std::cout << "i=" << ind <<std::endl;
                    if(close_id != -1){
                        double lidar_t = backend_buffer_lidar[close_id].first->header.stamp.toSec();
                        //std::cout << "lidar_t=" << lidar_t <<std::endl;
                        if(lidar_t > v_imu_t[v_imu_t.size()-1]){
                            for(int id = 0; id < backend_buffer_imu_t.size(); id++){
                                if(lidar_t >= backend_buffer_imu_t[id]){
                                    v_gyr_lidar.emplace_back(backend_buffer_gyr[id]);
                                    v_acc_lidar.emplace_back(backend_buffer_acc[id]);
                                    v_imu_t_lidar.emplace_back(backend_buffer_imu_t[id]);
                                }
                                else{
                                    break;
                                }
                            }
                        }
                        lidar_info = backend_buffer_lidar[close_id];
                        backend_buffer_lidar.erase(backend_buffer_lidar.begin(), backend_buffer_lidar.begin() + close_id + 1);
                    }
                    else{
                        backend_buffer_lidar.erase(backend_buffer_lidar.begin(), backend_buffer_lidar.end()-1);
                    }
                }

                return true;
            }
        });
        lock.unlock();
        backend_busy = true;
        //std::cout << "lidar_info size=" << lidar_info.first.data.size() <<std::endl;
        feature_tracker->ExtractFAST(feat_frame);
        PubTrackingImg(feat_frame);
        StereoMatcher::FramePtr stereo_frame = stereo_matcher->Process(feat_frame, img_r);
        BackEnd::FramePtr back_frame = Converter::Convert(feat_frame, cam_m, stereo_frame, cam_s,
                                                          v_gyr, v_acc, v_imu_t);
        back_frame->td = td;
        backend->ProcessFrame(back_frame, lidar_info, start_this_frame, v_acc_lidar, v_gyr_lidar, v_imu_t_lidar);

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

void System::RelocProcess() {
    while(1) {
        std::vector<Relocalization::FramePtr> v_reloc_frames;
        std::unique_lock<std::mutex> lock(mtx_reloc);
        cv_reloc.wait(lock, [&] {
            int end_index = -1;
            for(auto& pair : reloc_buffer_keyframe) {
                auto& back_frame = pair.first;
                auto& v_x3Dc = pair.second;
                while(!reloc_buffer_frame.empty()) {
                    if(back_frame == reloc_buffer_frame.front().second) {
                        auto& feat_frame = reloc_buffer_frame.front().first;
                        v_reloc_frames.emplace_back(Converter::Convert(feat_frame, back_frame, v_x3Dc));
                        reloc_buffer_frame.pop_front();
                        break;
                    }
                    else {
                        reloc_buffer_frame.pop_front();
                    }
                }
            }
            reloc_buffer_keyframe.clear();
            return (!v_reloc_frames.empty());
        });
        lock.unlock();

        for(auto& frame : v_reloc_frames) {
            reloc->ProcessFrame(frame);
        }
    }
}

void System::PubTrackingImg(FeatureTracker::FramePtr feat_frame) {
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

        for(auto& pub : pub_tracking_img) {
            pub(feat_frame->timestamp, tracking_img);
        }
    }
}
