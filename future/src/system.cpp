#include "system.h"
#include <glog/logging.h>

System::System(const std::string& config_file) {
    reset_flag = false;
    param = ConfigLoader::Load(config_file);
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

    feature_tracker = std::make_shared<FeatureTracker>(cam_m);
    stereo_matcher = std::make_shared<StereoMatcher>(cam_m, cam_s, param.p_rl[0], param.q_rl[0]);
}

System::~System() {}

void System::Reset() {
    reset_flag = true;
    LOG(WARNING) << "system reset flag is turn on!";
}

void System::Process(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp,
                     const Eigen::VecVector3d& v_gyr,
                     const Eigen::VecVector3d& v_acc,
                     const std::vector<double>& v_imu_timestamp)
{
    if(reset_flag) {
        LOG(WARNING) << "reset the system...";
        b_first_frame = true;
    }

    FeatureTracker::FramePtr feat_frame;
    if(b_first_frame) {
        feat_frame = feature_tracker->InitFirstFrame(img_l, timestamp);
        b_first_frame = false;
    }
    else {
        feat_frame = feature_tracker->Process(img_l, timestamp);
    }

    {
        std::map<uint64_t, cv::Point2f> m_id_history_tmp;
        std::map<uint64_t, std::shared_ptr<std::deque<cv::Point2f>>> m_id_optical_flow_tmp;
        for(int i = 0, n = feat_frame->pt_id.size(); i < n; ++i) {
            auto it = m_id_history.find(feat_frame->pt_id[i]);
            if(it != m_id_history.end()) {
                m_id_history_tmp[it->first] = it->second;
                auto d_pt = m_id_optical_flow[it->first];
                m_id_optical_flow_tmp[it->first] = d_pt;
                d_pt->emplace_back(feat_frame->pt[i]);
                while(d_pt->size() > 5) {
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
    }

    if(draw_tracking_img) {
        cv::Mat tracking_img;
        cv::cvtColor(feat_frame->img, tracking_img, CV_GRAY2BGR);

        for(auto& it : m_id_optical_flow) {
            uint64_t pt_id = it.first;
            std::deque<cv::Point2f>& history_pts = *it.second;
            for(int i = 0, n = history_pts.size() - 1; i < n; ++i) {
                cv::line(tracking_img, history_pts[i], history_pts[i+1], cv::Scalar(0, 255, 255), 2);
            }
            cv::circle(tracking_img, history_pts.back(), 4, cv::Scalar(0, 255, 0), -1);
        }

        draw_tracking_img(tracking_img, feat_frame->id, feat_frame->timestamp);
    }
}

void System::SetDrawTrackingImgCallback(std::function<void(const cv::Mat&, uint64_t, double)> callback) {
    draw_tracking_img = callback;
}
