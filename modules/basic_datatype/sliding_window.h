#pragma once
#include <memory>
#include <mutex>
#include "frame.h"
#include "mappoint.h"
#include "util_datatype.h"

class SlidingWindow {
public:
    SlidingWindow(int max_lens = 10);
    ~SlidingWindow();

    void clear();
    void push_back(const FramePtr& keyframe);
    std::vector<FramePtr> get() const;
    size_t size() const;
    bool empty() const;
    FramePtr back() const;

    void push_back(const MapPointPtr& mp);
    std::vector<MapPointPtr> get_mps();
private:
    int mMaxLens;
    std::deque<FramePtr> mdKeyFrames;
    mutable std::mutex mDequeueMutex;

    std::deque<MapPointWPtr> mdMapPoints;
    std::mutex mVecMapPtsMutex;
};

SMART_PTR(SlidingWindow);
