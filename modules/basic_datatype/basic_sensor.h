#pragma once
#include <memory>
#include <sophus/se3.hpp>

class SensorBase {
public:
    SensorBase() {}
    virtual ~SensorBase() {}

    Sophus::SE3d mTbs;
};

using SensorBasePtr = std::shared_ptr<SensorBase>;
using SensorBaseConstPtr = std::shared_ptr<const SensorBase>;

class ImuSensor : public SensorBase
{
public:
    ImuSensor() {}
    virtual ~ImuSensor() {}

    // Process noise
    double gyro_noise;
    double acc_noise;
    double gyro_bias_noise;
    double acc_bias_noise;
};

using ImuSensorPtr = std::shared_ptr<ImuSensor>;
using ImuSensorConstPtr = std::shared_ptr<const ImuSensor>;
