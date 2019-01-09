#pragma once
#include "util.h"
#include "camera.h"

class FeatureTracker {
public:
    FeatureTracker(CameraPtr camera_);
    ~FeatureTracker();

    struct Frame {
        uint64_t id;
        std::vector<uint> pt_track_count;
        std::vector<uint64_t> pt_id;
        std::vector<cv::Point2f> pt;

        cv::Mat img;
        cv::Mat clahe;
        ImagePyr img_pyr_grad;
        double timestamp;
    };
    SMART_PTR(Frame)

    FramePtr InitFirstFrame(const cv::Mat& img, double timestamp);
    FramePtr Process(const cv::Mat& img, double timestamp);
private:
    FramePtr InitFrame(const cv::Mat& img, double timestamp);
    void ExtractFAST(FramePtr frame);
    void TrackFeatures(FramePtr ref_frame, FramePtr cur_frame);

    uint64_t next_frame_id = 0;
    uint64_t next_pt_id = 0;
    CameraPtr camera;
    FramePtr last_frame;
    cv::Ptr<cv::CLAHE> clahe;

    int fast_threshold = 20;
    int feat_min_dist = 32;
    float reproj_threshold = 3.0f;
};
SMART_PTR(FeatureTracker)
