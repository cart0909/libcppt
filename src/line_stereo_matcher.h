#pragma once
#include "util.h"
#include "line_tracker.h"

class LineStereoMatcher {
public:
    LineStereoMatcher();

    struct Frame {
        // size is same as LineTracker::Frame::v_lines
        cv::Mat img_r;
        std::vector<cv::line_descriptor::KeyLine> v_lines_r;
    };
    SMART_PTR(Frame)

    FramePtr Process(LineTracker::FramePtr l_frame, const cv::Mat& img_r);
private:
    std::vector<cv::line_descriptor::KeyLine> DetectLineFeatures(const cv::Mat& img);
    int Match(const cv::Mat& desc1, const cv::Mat& desc2, float nnr, std::vector<int>& matches12);
    int LRMatch(const cv::Mat& desc1, const cv::Mat& desc2, float nnr, std::vector<int>& matches12);
    cv::Ptr<cv::line_descriptor::BinaryDescriptor> lbd;
    cv::Ptr<cv::ximgproc::FastLineDetector> fld;
};
SMART_PTR(LineStereoMatcher)
