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
        std::vector<cv::Point2f> pt_l;
        std::vector<cv::Point2f> pt_r; // if pt_r[i][0] == -1 -> pt[i] is mono meas.
        uint num_stereo;

        cv::Mat img_l, img_r;
        cv::Mat clahe_l, clahe_r;
        ImagePyr img_pyr_grad_l, img_pyr_grad_r;
        double timestamp;

        bool b_keyframe = false;
    };
    SMART_PTR(Frame)

    FramePtr InitFirstFrame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp);
    FramePtr Process(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp);
private:

    FramePtr InitFrame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp);
    void ExtractFAST(FramePtr frame);
    void TrackFeatures(FramePtr ref_frame, FramePtr cur_frame);

    uint64_t next_frame_id = 0;
    uint64_t next_pt_id = 0;
    CameraPtr camera;
    FramePtr last_frame;
    cv::Ptr<cv::CLAHE> clahe;

    int fast_threshold = 20;
    int feat_min_dist = 32;
};

SMART_PTR(FeatureTracker)
