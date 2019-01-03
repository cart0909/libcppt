#pragma once
#include "basic_datatype/util_datatype.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"
#include <thread>

class BackEnd {
public:
    BackEnd(SimpleStereoCamPtr camera_, SlidingWindowPtr sliding_window_);
    void Process();

    SimpleStereoCamPtr camera;
    SlidingWindowPtr sliding_window;

    // buffer
};

SMART_PTR(BackEnd)
