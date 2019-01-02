#pragma once
#include <string>
#include <thread>
#include "basic_datatype/imu_data.h"
#include "basic_datatype/util_datatype.h"

class VIOSystem {
public:
    VIOSystem();
    ~VIOSystem();

    void ProcessIMU(const std::vector<ImuData>& v_imu_data);
    void ProcessImage(const cv::Mat& img_raw_l, const cv::Mat& img_raw_r, double timestamp);

private:

};

SMART_PTR(VIOSystem)
