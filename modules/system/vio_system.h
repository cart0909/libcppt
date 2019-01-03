#pragma once
#include <string>
#include <thread>
#include "basic_datatype/imu_data.h"
#include "basic_datatype/frame.h"
#include "basic_datatype/util_datatype.h"
#include "camera_model/simple_stereo_camera.h"
#include "front_end/simple_frontend.h"
#include "front_end/imu_preintegration.h"
#include "back_end/backend.h"

class VIOSystem {
public:
    VIOSystem(const std::string& config_file);
    ~VIOSystem();

    void Process(const cv::Mat& img_raw_l, const cv::Mat& img_raw_r, double timestamp,
                 const std::vector<ImuData>& v_imu_data);

private:
    FramePtr cur_frame, last_frame;
    ImuSensorPtr imu_sensor;
    SimpleStereoCamPtr camera;
    SimpleFrontEndPtr front_end;
    BackEndPtr back_end;
    SlidingWindowPtr sliding_window;
    std::thread t_backend;

    Eigen::Matrix<double, 6, 6> gyr_acc_cov;
    double gravity_magnitude;
    double last_imu_timestamp;
    ImuPreintegrationPtr imu_preintegration;
};

SMART_PTR(VIOSystem)
