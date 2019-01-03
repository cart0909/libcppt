#pragma once
#include "basic_datatype/util_datatype.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"
#include <thread>
#include <condition_variable>

class BackEnd {
public:
    enum State {
        NEED_INIT,
        CV_ONLY,
        TIGHTLY_COUPLE
    };

    BackEnd(SimpleStereoCamPtr camera_, SlidingWindowPtr sliding_window_);
    ~BackEnd();

    void Process();
    void AddKeyFrame(FramePtr keyframe);

    State state;

    SimpleStereoCamPtr camera;
    SlidingWindowPtr sliding_window;

    // buffer
    std::vector<FramePtr> v_kf_buffer;
    std::mutex m_kf_buffer;
    std::condition_variable cv_kf_buffer;
private:
    void ProcessKeyFrame(FramePtr keyframe);
};

SMART_PTR(BackEnd)
