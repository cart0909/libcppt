#pragma once
#include <atomic>
#include "util.h"
#include "config_loader.h"
#include "feature_tracker.h"
#include "stereo_matcher.h"
#include "backend.h"
#include "relocalization.h"
#include "pose_faster.h"

class System {
public:
    System();
    System(const std::string& config_file);
    virtual ~System();

    void Reset();
    void PushImages(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp);
    void PushImuData(const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double timestamp);

    inline void SubTrackingImg(std::function<void(double, const cv::Mat&)> callback) {
        pub_tracking_img.emplace_back(callback);
    }

    inline void SubMapPoints(std::function<void(const Eigen::VecVector3d&)> callback) {
        backend->SubMapPoints(callback);
    }

    inline void SubVIOTwc(std::function<void(double, const Sophus::SE3d&)> callback) {
        backend->SubVIOTwc(callback);
    }

    inline void SubRelocTwc(std::function<void(double, const Sophus::SE3d&)> callback) {
        if(reloc)
            reloc->SubRelocTwc(callback);
    }

    inline void SubAddRelocPath(std::function<void(const Sophus::SE3d&)> callback) {
        if(reloc)
            reloc->SubAddRelocPath(callback);
    }

    inline void SubUpdateRelocPath(std::function<void(const std::vector<Sophus::SE3d>&)> callback) {
        if(reloc)
            reloc->SubUpdateRelocPath(callback);
    }

    inline void SubRelocImg(std::function<void(const cv::Mat&)> callback) {
        if(reloc)
            reloc->SubRelocImg(callback);
    }

    inline void SubLoopEdge(std::function<void(const std::pair<uint64_t, uint64_t>&)> callback) {
        if(reloc)
            reloc->SubLoopEdge(callback);
    }

    inline bool Predict(const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double t, Sophus::SE3d& Twc) {
        if(pose_faster)
            return pose_faster->Predict(gyr, acc, t, Twc);
        return false;
    }

protected:
    virtual void InitCameraParameters();
    virtual void FrontEndProcess();
    virtual void BackEndProcess();
    void RelocProcess();
    void PubTrackingImg(FeatureTracker::FramePtr feat_frame);

    std::atomic<bool> reset_flag;
    bool b_first_frame = true;
    ConfigLoader::Param param;
    CameraPtr cam_m; // camera master
    CameraPtr cam_s; // camera slave

    FeatureTrackerPtr feature_tracker;
    std::thread frontend_thread;
    std::mutex mtx_frontend;
    std::condition_variable cv_frontend;
    std::deque<std::tuple<cv::Mat, cv::Mat, double>> frontend_buffer_img;

    std::atomic<bool> backend_busy;
    StereoMatcherPtr stereo_matcher;
    BackEndPtr backend;
    std::thread backend_thread;
    std::mutex mtx_backend;
    std::condition_variable cv_backend;
    std::deque<std::pair<FeatureTracker::FramePtr, cv::Mat>> backend_buffer_img;
    Eigen::DeqVector3d backend_buffer_gyr;
    Eigen::DeqVector3d backend_buffer_acc;
    std::deque<double> backend_buffer_imu_t;

    RelocalizationPtr reloc;
    std::thread reloc_thread;
    std::mutex mtx_reloc;
    std::condition_variable cv_reloc;
    std::deque<std::pair<FeatureTracker::FramePtr, BackEnd::FramePtr>> reloc_buffer_frame;
    std::deque<std::pair<BackEnd::FramePtr, Eigen::VecVector3d>> reloc_buffer_keyframe;

    PoseFasterPtr pose_faster;

    std::map<uint64_t, cv::Point2f> m_id_history;
    std::map<uint64_t, std::shared_ptr<std::deque<cv::Point2f>>> m_id_optical_flow;
    std::vector<std::function<void(double, const cv::Mat&)>> pub_tracking_img;
    Eigen::VecVector3d v_cache_gyr, v_cache_acc;
    std::vector<double> v_cache_imu_timestamps;
};
SMART_PTR(System)
