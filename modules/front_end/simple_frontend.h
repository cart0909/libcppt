#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "basic_datatype/frame.h"
#include "basic_datatype/util_datatype.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"

class SimpleFrontEnd {
public:
    SimpleFrontEnd(const SimpleStereoCamPtr& camera,
                   const SlidingWindowPtr& sliding_window);
    ~SimpleFrontEnd();

    // track features by optical flow and check epipolar constrain
    void TrackFeaturesByOpticalFlow(const FrameConstPtr& ref_frame,
                                    const FramePtr& cur_frame);
    void TrackFeatLKWithEstimateTcr(const FrameConstPtr& ref_frame,
                                    const FramePtr& cur_frame,
                                    const Sophus::SE3d& Tcr);
    void PoseOpt(const FramePtr& frame, const Sophus::SE3d& init_Twc);
private:
    void RemoveOutlierFromF(std::vector<cv::Point2f>& ref_pts,
                            const FramePtr& cur_frame);

    void UniformFeatureDistribution(const FramePtr& cur_frame);

    SimpleStereoCamPtr mpCamera;
    SlidingWindowPtr mpSldingWindow;
};

SMART_PTR(SimpleFrontEnd)
