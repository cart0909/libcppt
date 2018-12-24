#pragma once
#include <vector>
#include <mutex>
#include <functional>
#include <condition_variable>
#include "basic_datatype/frame.h"
#include "basic_datatype/sliding_window.h"
#include "camera_model/simple_stereo_camera.h"
#include "basic_datatype/util_datatype.h"
#include "ceres/marginalization_factor.h"

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
    void SetDebugCallback(const std::function<void(const std::vector<Sophus::SE3d>&,
                                                   const VecVector3d&)>& callback);
    BackEndState mState;
private:
    bool InitSystem(const FramePtr& keyframe);
    void CreateMapPoint(const FramePtr& keyframe);
    void ShowResultGUI() const;
    void SlidingWindowBA(const FramePtr& new_keyframe);
    void Marginalization(const FramePtr& new_keyframe);

    SimpleStereoCamPtr mpCamera;
    SlidingWindowPtr mpSlidingWindow;

    // buffer
    std::vector<FramePtr> mKFBuffer;
    std::mutex mKFBufferMutex;
    std::condition_variable mKFBufferCV;

    // callback function
    std::function<void(const std::vector<Sophus::SE3d>&,
                       const VecVector3d&)> mDebugCallback;

    // marginalization
    MarginalizationInfo* mpMarginInfo = nullptr;
    std::vector<double*> mvMarginParameterBlock;
};

SMART_PTR(SimpleBackEnd)
