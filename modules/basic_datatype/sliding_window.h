#pragma once
#include <memory>
#include <mutex>
#include "frame.h"
#include "util_datatype.h"

class SlidingWindow {
public:
    SlidingWindow(int max_lens = 10);
    ~SlidingWindow();

    void clear();
    void push_back(const FramePtr& keyframe);
    std::vector<FramePtr> get() const;
    size_t size() const;
private:
    int mMaxLens;
    std::deque<FramePtr> mdKeyFrames;
    mutable std::mutex mDequeueMutex;
};

SMART_PTR(SlidingWindow);
