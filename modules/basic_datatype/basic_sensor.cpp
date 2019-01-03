#include "basic_sensor.h"

SensorBase::SensorBase() {}
SensorBase::SensorBase(const Sophus::SE3d& Tbs_) : Tbs(Tbs_) {}
SensorBase::~SensorBase() {}

ImuSensor::ImuSensor() {}
ImuSensor::ImuSensor(const Sophus::SE3d& Tbs,
                     double gn, double an, double gbn, double abn)
    : SensorBase(Tbs), gyr_noise(gn), acc_noise(an), gyr_bias_noise(gbn), acc_bias_noise(abn) {
    gyr_noise_cov = gyr_noise * gyr_noise * Eigen::Matrix3d::Identity();
    acc_noise_cov = acc_noise * acc_noise * Eigen::Matrix3d::Identity();
    gyr_bias_cov = gyr_bias_noise * gyr_bias_noise * Eigen::Matrix3d::Identity();
    acc_bias_cov = acc_bias_noise * acc_bias_noise * Eigen::Matrix3d::Identity();
}
ImuSensor::~ImuSensor() {}
