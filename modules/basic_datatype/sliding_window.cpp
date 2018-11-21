#include "sliding_window.h"

SlidingWindow::SlidingWindow(int max_lens)
    : mMaxLens(max_lens) {}
SlidingWindow::~SlidingWindow() {}

void SlidingWindow::clear() {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    mdKeyFrames.clear();
}

void SlidingWindow::push_back(const FramePtr& keyframe) {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    if(mdKeyFrames.size() == mMaxLens) {
        mdKeyFrames.pop_front();
    }
    mdKeyFrames.push_back(keyframe);
}

std::vector<FramePtr> SlidingWindow::get() const {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    return std::vector<FramePtr>(mdKeyFrames.begin(), mdKeyFrames.end());
}

size_t SlidingWindow::size() const {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    return mdKeyFrames.size();
}
