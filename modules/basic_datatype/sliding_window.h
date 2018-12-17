#pragma once
#include <memory>
#include <mutex>
#include "frame.h"
#include "mappoint.h"
#include "util_datatype.h"

class SlidingWindow {
public:
    SlidingWindow(int max_len_ = 10);

    void clear_all();

    void push_kf(FramePtr keyframe);
    void push_mp(MapPointPtr mappoint);
    void push_mps(const std::vector<MapPointPtr>& mps);

    size_t size_kfs() const;
    size_t size_mps() const;

    std::vector<FramePtr> get_kfs() const;
    FramePtr front_kf() const;
    std::vector<MapPointPtr> get_mps();

    std::mutex big_mutex;
    const int max_len;
private:
    mutable std::mutex kfs_mutex;
    mutable std::mutex mps_mutex;
    std::deque<FramePtr> kfs_buffer;
    std::deque<MapPointWPtr> mps_buffer;
};
SMART_PTR(SlidingWindow);
