#pragma once
#include <string>
#include <memory>
#include <thread>
#include "camera_model/simple_stereo_camera.h"
#include "front_end/simple_frontend.h"
#include "back_end/isam2_backend.h"

class VOSystem {
public:
    VOSystem(const std::string& config_file);
    ~VOSystem();

    void Process(const cv::Mat& img_raw_l, const cv::Mat& img_raw_r);
private:
    SimpleStereoCamPtr mpStereoCam;
    SimpleFrontEndPtr  mpFrontEnd;
    ISAM2BackEndPtr    mpBackEnd;
    FramePtr mpLastFrame;

    std::thread mtBackEnd;
};

using VOSystemPtr = std::shared_ptr<VOSystem>;
using VOSystemConstPtr = std::shared_ptr<const VOSystem>;
