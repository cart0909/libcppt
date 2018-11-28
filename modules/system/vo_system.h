#pragma once
#include <string>
#include <memory>
#include <thread>
#include "basic_datatype/util_datatype.h"
#include "camera_model/simple_stereo_camera.h"
#include "front_end/simple_frontend.h"
#include "back_end/simple_backend.h"
#include "basic_datatype/sliding_window.h"

class VOSystem {
public:
    VOSystem(const std::string& config_file);
    ~VOSystem();

    void Process(const cv::Mat& img_raw_l, const cv::Mat& img_raw_r,
                 double timestamp);

    SimpleStereoCamPtr mpStereoCam;
    SimpleFrontEndPtr  mpFrontEnd;
    SimpleBackEndPtr   mpBackEnd;
    SlidingWindowPtr   mpSlidingWindow;
    FramePtr mpLastFrame;

    std::thread mtBackEnd;

    std::function<void(const Sophus::SE3d, double timestamp)> mDebugCallback;
};

SMART_PTR(VOSystem)
