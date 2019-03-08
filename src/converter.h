#pragma once
#include "util.h"
#include "feature_tracker.h"
#include "stereo_matcher.h"
#include "backend.h"
#include "relocalization.h"

class Converter {
public:
    // stereo camera
    static BackEnd::FramePtr Convert(FeatureTracker::FramePtr feat_frame, CameraPtr cam_master,
                                     StereoMatcher::FramePtr stereo_frame, CameraPtr cam_slave,
                                     const Eigen::VecVector3d& v_gyr, const Eigen::VecVector3d& v_acc,
                                     const std::vector<double>& v_imu_timestamp);

    // rgbd camera
    static BackEnd::FramePtr Convert(FeatureTracker::FramePtr feat_frame, CameraPtr camera,
                                     const cv::Mat& depth_iamge, double depth_units,
                                     const Sophus::SO3d& q_rl, const Eigen::Vector3d& p_rl,
                                     const Eigen::VecVector3d& v_gyr, const Eigen::VecVector3d& v_acc,
                                     const std::vector<double>& v_imu_timestamp);

    static Relocalization::FramePtr Convert(FeatureTracker::FramePtr feat_frame,
                                            BackEnd::FramePtr back_frame,
                                            const Eigen::VecVector3d& v_x3Dc);
};
