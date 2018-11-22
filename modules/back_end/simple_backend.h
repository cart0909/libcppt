#pragma once
#include <vector>
#include <mutex>
#include <condition_variable>
#include "basic_datatype/frame.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"
#include "basic_datatype/util_datatype.h"
#include <functional>

class SimpleBackEnd {
public:
    enum BackEndState {
        INIT,
        NON_LINEAR
    };

    SimpleBackEnd(const SimpleStereoCamPtr& camera,
                  const SlidingWindowPtr& sliding_window);
    ~SimpleBackEnd();

    void Process();
    void AddKeyFrame(const FramePtr& keyframe);
private:
    BackEndState mState;
    SimpleStereoCamPtr mpCamera;
    SlidingWindowPtr mpSlidingWindow;

    // buffer
    std::vector<FramePtr> mKFBuffer;
    std::mutex mKFBufferMutex;
    std::condition_variable mKFBufferCV;
};

SMART_PTR(SimpleBackEnd)
