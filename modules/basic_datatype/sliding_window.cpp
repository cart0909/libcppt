#include "sliding_window.h"

SlidingWindow::SlidingWindow(int max_len_)
    : max_len(max_len_) {}

void SlidingWindow::clear_all() {
    std::unique_lock<std::mutex> lock0(kfs_mutex), lock1(mps_mutex);
    kfs_buffer.clear();
    mps_buffer.clear();
}

void SlidingWindow::push_kf(FramePtr keyframe) {
    std::unique_lock<std::mutex> lock(kfs_mutex);
    while(kfs_buffer.size() >= max_len) {
        kfs_buffer.pop_front();
    }
    kfs_buffer.push_back(keyframe);
}

void SlidingWindow::push_mp(MapPointPtr mappoint) {
    std::unique_lock<std::mutex> lock(mps_mutex);
    mps_buffer.push_back(mappoint);
}

size_t SlidingWindow::size_kfs() const {
    std::unique_lock<std::mutex> lock(kfs_mutex);
    return kfs_buffer.size();
}

size_t SlidingWindow::size_mps() const {
    std::unique_lock<std::mutex> lock(mps_mutex);
    return mps_buffer.size();
}

std::vector<FramePtr> SlidingWindow::get_kfs() const {
    std::unique_lock<std::mutex> lock(kfs_mutex);
    return std::vector<FramePtr>(kfs_buffer.begin(), kfs_buffer.end());
}

std::vector<MapPointPtr> SlidingWindow::get_mps() {
    std::vector<MapPointPtr> temp;
    std::unique_lock<std::mutex> lock(mps_mutex);
    for(auto it = mps_buffer.begin(); it != mps_buffer.end();) {
        if(it->expired()) {
            it = mps_buffer.erase(it);
        }
        else {
            temp.emplace_back(it->lock());
            ++it;
        }
    }
    return temp;
}
