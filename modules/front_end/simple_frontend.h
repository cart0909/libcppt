#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "basic_datatype/frame.h"
#include "camera_model/simple_stereo_camera.h"

class SimpleFrontEnd {
public:
    SimpleFrontEnd(const SimpleStereoCamPtr& camera);
    ~SimpleFrontEnd();

    // with frame without any feature extract by FAST corner detector
    // if exist some features extract FAST corner in empty grid.
    void ExtractFeatures(FramePtr frame);

    // track features by optical flow and check epipolar constrain
    void TrackFeaturesByOpticalFlow(FramePtr ref_frame, FramePtr cur_frame);

    // simple sparse stereo matching algorithm by optical flow
    void SparseStereoMatching(FramePtr frame);

private:
    void RemoveOutlierFromF(std::vector<uint64_t>& ref_pt_id,
                            std::vector<uint32_t>& ref_pt_count,
                            std::vector<cv::Point2f>& ref_pts,
                            std::vector<cv::Point2f>& cur_pts);

    void UniformFeatureDistribution(std::vector<uint64_t>& ref_pt_id,
                                    std::vector<uint32_t>& ref_pt_count,
                                    std::vector<cv::Point2f>& cur_pts);

    SimpleStereoCamPtr mpCamera;
    uint64_t mFeatureID;
};

using SimpleFrontEndPtr = std::shared_ptr<SimpleFrontEnd>;
using SimpleFrontEndConstPtr = std::shared_ptr<const SimpleFrontEnd>;
