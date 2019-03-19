#pragma once
#include "system.h"
#include "line_tracker.h"
#include "line_stereo_matcher.h"

class PLSystem : public System
{
public:
    PLSystem();
    PLSystem(const std::string& config_file);
    ~PLSystem();
private:
    void FrontEndProcess() override;
    void BackEndProcess() override;
    void PubTrackingImg(FeatureTracker::FramePtr feat_frame, LineTracker::FramePtr line_frame);
    LineTrackerPtr line_tracker;
    LineStereoMatcherPtr line_stereo_matcher;

    std::deque<std::tuple<FeatureTracker::FramePtr, LineTracker::FramePtr, cv::Mat>> backend_buffer_img;
};