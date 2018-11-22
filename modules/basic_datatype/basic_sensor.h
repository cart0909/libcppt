#pragma once
#include <memory>
#include <sophus/se3.hpp>
#include "basic_datatype/util_datatype.h"

class SensorBase {
public:
    SensorBase() {}
    SensorBase(const Sophus::SE3d& Tbs_) : Tbs(Tbs_) {}
    virtual ~SensorBase() {}

    Sophus::SE3d Tbs;
};

SMART_PTR(SensorBase)

class ImuSensor : public SensorBase
{
public:
    ImuSensor() {}
    ImuSensor(double gn, double an, double gbn, double abn)
        :gyro_noise(gn), acc_noise(an), gyro_bias_noise(gbn), acc_bias_noise(abn) {}
    virtual ~ImuSensor() {}

    // Process noise
    double gyro_noise;
    double acc_noise;
    double gyro_bias_noise;
    double acc_bias_noise;
};

SMART_PTR(ImuSensor)
