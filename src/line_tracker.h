#pragma once
#include "util.h"
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <opencv2/ximgproc.hpp>
#include "camera.h"

class LineTracker {
public:
    LineTracker(CameraPtr camera_);

    struct Frame {
        uint64_t frame_id;
        cv::Mat img;
        cv::Mat debug_img;
        double timestamp;
        std::vector<uint64_t> v_line_id;
        std::vector<cv::line_descriptor::KeyLine> v_lines;
        cv::Mat desc;
    };
    SMART_PTR(Frame)

    FramePtr InitFirstFrame(const cv::Mat& img, double timestamp);
    FramePtr Process(const cv::Mat& img, double timestamp);
private:
    FramePtr InitFrame(const cv::Mat& img, double timestamp);
    int Match(const cv::Mat& desc1, const cv::Mat& desc2, float nnr, std::vector<int>& matches12);
    int LRConsistencyMatch(const cv::Mat& desc1, const cv::Mat& desc2, float nnr, std::vector<int>& matches12);

    uint64_t next_frame_id = 0;
    uint64_t next_line_id = 0;
    FramePtr last_frame;

    cv::Ptr<cv::ximgproc::FastLineDetector> fld;
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd;
    CameraPtr camera;
};
SMART_PTR(LineTracker)
