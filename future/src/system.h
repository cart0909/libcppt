#pragma once
#include <atomic>
#include "util.h"
#include "config_loader.h"
#include "feature_tracker.h"
#include "stereo_matcher.h"
#include "backend.h"

class System {
public:
    System(const std::string& config_file);
    ~System();

    void Reset();
    void Process(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp,
                 const Eigen::VecVector3d& v_gyr,
                 const Eigen::VecVector3d& v_acc,
                 const std::vector<double>& v_imu_timestamp);

    void SetDrawTrackingImgCallback(std::function<void(const cv::Mat&, uint64_t, double)> callback);
    inline void SetDrawMapPointCallback(std::function<void(const Eigen::VecVector3d&)> callback) {
        backend->SetDrawMapPointCallback(callback);
    }
    inline void SetDrawPoseCallback(std::function<void(uint, double, const Sophus::SE3d&)> callback) {
        backend->SetDrawPoseCallback(callback);
    }
private:
    std::atomic<bool> reset_flag;
    bool b_first_frame = true;
    ConfigLoader::Param param;
    CameraPtr cam_m; // camera master
    CameraPtr cam_s; // camera slave
    FeatureTrackerPtr feature_tracker;
    StereoMatcherPtr stereo_matcher;
    BackEndPtr backend;

    std::map<uint64_t, cv::Point2f> m_id_history;
    std::map<uint64_t, std::shared_ptr<std::deque<cv::Point2f>>> m_id_optical_flow;

    std::function<void(const cv::Mat&, uint64_t, double)> draw_tracking_img;

    Eigen::VecVector3d v_cache_gyr, v_cache_acc;
    std::vector<double> v_cache_imu_timestamps;
};
SMART_PTR(System)
