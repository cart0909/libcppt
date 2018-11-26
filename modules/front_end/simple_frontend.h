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

    // with frame without any feature extract by FAST corner detector
    // if exist some features extract FAST corner in empty grid.
    void ExtractFeatures(const FramePtr& frame);

    // track features by optical flow and check epipolar constrain
    void TrackFeaturesByOpticalFlow(const FrameConstPtr& ref_frame,
                                    const FramePtr& cur_frame);

    // simple sparse stereo matching algorithm by optical flow
    void SparseStereoMatching(const FramePtr& frame);

    void PoseOpt(const FramePtr& frame);
private:
    void RemoveOutlierFromF(std::vector<cv::Point2f>& ref_pts,
                            const FramePtr& cur_frame);

    void UniformFeatureDistribution(const FramePtr& cur_frame);

    SimpleStereoCamPtr mpCamera;
    SlidingWindowPtr mpSldingWindow;
    uint64_t mFeatureID;
};

SMART_PTR(SimpleFrontEnd)
