#include "stereo_camera.h"
#include <ros/ros.h>

StereoCamera::StereoCamera(const ImuSensorPtr& imu_sensor,
             const PinholeCameraPtr& left_cam,
             const PinholeCameraPtr& right_cam)
    : mpImuSensor(imu_sensor), mpCamera{left_cam, right_cam}
{
    ROS_ASSERT(mpCamera[0] && mpCamera[1]);
    if(mpImuSensor)
        mbNoImuSensor = false;
    else
        mbNoImuSensor = true;

    // mTij[I_IMU][I_IMU] = mTij[I_LC][I_LC] = mTij[I_RC][I_RC]
    mTij[I_IMU][I_LC] = mpImuSensor->mTbs.inverse() * mpCamera[I_LC]->mTbs;
    mTij[I_IMU][I_RC] = mpImuSensor->mTbs.inverse() * mpCamera[I_RC]->mTbs;
    mTij[I_LC][I_IMU] = mTij[I_IMU][I_LC].inverse();
    mTij[I_RC][I_IMU] = mTij[I_IMU][I_RC].inverse();
    mTij[I_LC][I_RC] = mpCamera[I_LC]->mTbs.inverse() *  mpCamera[I_RC]->mTbs;
    mTij[I_RC][I_LC] = mTij[I_LC][I_RC].inverse();
}
