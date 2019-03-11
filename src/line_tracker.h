#pragma once
#include "util.h"

class LineTracker {
public:
    struct Frame {
        uint64_t frame_id;
        cv::Mat rect_img;
        double timestamp;
    };
    SMART_PTR(Frame)

private:
    FramePtr InitFrame(const cv::Mat& rect_img, double timestamp) {
        FramePtr frame(new Frame);
        frame->frame_id = next_frame_id++;
        frame->rect_img = rect_img;
        frame->timestamp = timestamp;
        return frame;
    }

    uint64_t next_frame_id = 0;
    FramePtr last_frame;
};
SMART_PTR(LineTracker)
