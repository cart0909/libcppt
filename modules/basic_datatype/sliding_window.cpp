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
    mdKeyFrames.emplace_back(keyframe);
}

std::vector<FramePtr> SlidingWindow::get() const {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    return std::vector<FramePtr>(mdKeyFrames.begin(), mdKeyFrames.end());
}

size_t SlidingWindow::size() const {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    return mdKeyFrames.size();
}

bool SlidingWindow::empty() const {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    return mdKeyFrames.empty();
}

FramePtr SlidingWindow::back() const {
    std::unique_lock<std::mutex> lock(mDequeueMutex);
    return mdKeyFrames.back();
}

void SlidingWindow::push_back(const MapPointPtr& mp) {
    std::unique_lock<std::mutex> lock(mVecMapPtsMutex);
    mdMapPoints.emplace_back(mp);
}

std::vector<MapPointPtr> SlidingWindow::get_mps() {
    std::vector<MapPointPtr> v_mps;
    std::unique_lock<std::mutex> lock(mVecMapPtsMutex);

    for(auto it = mdMapPoints.begin(); it != mdMapPoints.end();) {
        if(it->expired())
            it = mdMapPoints.erase(it);
        else {
            auto mp = it->lock();
            if(mp->empty())
                it = mdMapPoints.erase(it);
            else {
                v_mps.emplace_back(mp);
                ++it;
            }
        }
    }
    return v_mps;
}
