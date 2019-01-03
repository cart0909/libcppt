#include "backend.h"
#include <ros/ros.h>

BackEnd::BackEnd(SimpleStereoCamPtr camera_, SlidingWindowPtr sliding_window_)
    : state(NEED_INIT), camera(camera_), sliding_window(sliding_window_) {}

BackEnd::~BackEnd() {}

void BackEnd::Process() {
    while(1) {
        std::vector<FramePtr> v_keyframe;
        std::unique_lock<std::mutex> lock(m_kf_buffer);
        cv_kf_buffer.wait(lock, [&]{
            v_keyframe = v_kf_buffer;
            return !v_keyframe.empty();
        });
        lock.unlock();

        for(auto& keyframe : v_keyframe) {
            ProcessKeyFrame(keyframe);
        }
    }
}

void BackEnd::AddKeyFrame(FramePtr keyframe) {
    m_kf_buffer.lock();
    v_kf_buffer.emplace_back(keyframe);
    m_kf_buffer.unlock();
    cv_kf_buffer.notify_one();
}

void BackEnd::ProcessKeyFrame(FramePtr keyframe) {
    if(state == NEED_INIT) {

    }
    else if(state == CV_ONLY) {

    }
    else if(state == TIGHTLY_COUPLE) {

    }
    else {
        ROS_ERROR_STREAM("[BackEnd] Error State!");
        exit(-1);
    }
}
