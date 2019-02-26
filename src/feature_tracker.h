#pragma once
#include "util.h"
#include "camera.h"

class FeatureTracker {
public:
    FeatureTracker(CameraPtr camera_, double clahe_parameter, int fast_threshold_,
                   int min_dist, float reproj_threshold_);
    ~FeatureTracker();

    struct Frame {
        uint64_t id;
        std::vector<uint> pt_track_count;
        std::vector<uint64_t> pt_id;
        std::vector<cv::Point2f> pt;

        cv::Mat img;
        cv::Mat compressed_img;
        cv::Mat clahe;
        ImagePyr img_pyr_grad;
        double timestamp;
    };
    SMART_PTR(Frame)

    FramePtr InitFirstFrame(const cv::Mat& img, double timestamp);
    FramePtr Process(const cv::Mat& img, double timestamp);
    void ExtractFAST(FramePtr frame);
private:
    FramePtr InitFrame(const cv::Mat& img, double timestamp);
    void TrackFeatures(FramePtr ref_frame, FramePtr cur_frame);

    uint64_t next_frame_id = 0;
    uint64_t next_pt_id = 0;
    CameraPtr camera;
    FramePtr last_frame;
    cv::Ptr<cv::CLAHE> clahe;

    int fast_threshold;
    int feat_min_dist;
    float reproj_threshold;
};
SMART_PTR(FeatureTracker)
