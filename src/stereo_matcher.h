#pragma once
#include "util.h"
#include "camera.h"
#include "feature_tracker.h"

class StereoMatcher {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    StereoMatcher(CameraPtr cam_l_, CameraPtr cam_r_, Eigen::Vector3d prl_, Sophus::SO3d qrl_,
                  double clahe_parameter, float dist_epipolar_threshold_);
    ~StereoMatcher();

    struct Frame {
        // size is same as FeatureTracker::Frame::pt_id
        std::vector<cv::Point2f> pt_r;
        cv::Mat img_r;
        cv::Mat clahe_r;
        ImagePyr img_pyr_grad_r;
    };
    SMART_PTR(Frame)

    FramePtr Process(FeatureTracker::FrameConstPtr feat_frame, const cv::Mat& img_r);
private:

    FramePtr InitFrame(const cv::Mat& img_r);

    CameraPtr cam_l, cam_r;
    Eigen::Vector3d prl;
    Sophus::SO3d qrl;
    cv::Ptr<cv::CLAHE> clahe;

    float dist_epipolar_threshold;
};
SMART_PTR(StereoMatcher)
