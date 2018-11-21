#pragma once
#include "basic_datatype/frame.h"
#include "camera_model/simple_stereo_camera.h"
#include <functional>

class SimpleBackEnd {
public:
    SimpleBackEnd();
    ~SimpleBackEnd();

    void Process();
    void AddKeyFrame(const FramePtr& keyframe);
private:
};
