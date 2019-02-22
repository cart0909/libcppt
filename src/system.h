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
    System(const std::string& config_file);
    ~System();

    void Reset();
    void Process(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp,
                 const Eigen::VecVector3d& v_gyr,
                 const Eigen::VecVector3d& v_acc,
                 const std::vector<double>& v_imu_timestamp);

    inline void SubTrackingImg(std::function<void(double, const cv::Mat&)> callback) {
        pub_tracking_img.emplace_back(callback);
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
        return  false;
    }

private:
    void PushKeyFrame2Reloc(BackEnd::FramePtr back_frame, const Eigen::VecVector3d& v_x3Dc);

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

    std::vector<std::function<void(double, const cv::Mat&)>> pub_tracking_img;

    Eigen::VecVector3d v_cache_gyr, v_cache_acc;
    std::vector<double> v_cache_imu_timestamps;

    std::mutex mtx_reloc_cache;
    std::deque<std::pair<FeatureTracker::FramePtr, BackEnd::FramePtr>> d_reloc_cache;
    RelocalizationPtr reloc;

    PoseFasterPtr pose_faster;
};
SMART_PTR(System)
