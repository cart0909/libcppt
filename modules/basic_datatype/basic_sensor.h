#pragma once
#include <memory>
#include <sophus/se3.hpp>
#include "basic_datatype/util_datatype.h"

class SensorBase {
public:
    SensorBase();
    SensorBase(const Sophus::SE3d& Tbs_);
    virtual ~SensorBase();

    Sophus::SE3d Tbs;
};
SMART_PTR(SensorBase)

class ImuSensor : public SensorBase
{
public:
    ImuSensor();
    ImuSensor(const Sophus::SE3d& Tbs,
              double gn, double an, double gbn, double abn);
    virtual ~ImuSensor();

    // Process noise
    double gyr_noise;
    double acc_noise;
    double gyr_bias_noise;
    double acc_bias_noise;

    Eigen::Matrix3d gyr_noise_cov;
    Eigen::Matrix3d acc_noise_cov;
    Eigen::Matrix3d gyr_bias_cov; // random walk
    Eigen::Matrix3d acc_bias_cov; // random walk
};
SMART_PTR(ImuSensor)
