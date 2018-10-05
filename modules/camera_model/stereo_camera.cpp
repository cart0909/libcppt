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

    Eigen::Matrix3d Rrl = mTij[I_RC][I_LC].rotationMatrix();
    Eigen::Matrix3d trl_hat = Sophus::SO3d::hat(mTij[I_RC][I_LC].translation());
    mE = Rrl * trl_hat;
    mE /= mE(2, 2);

    mF = mpCamera[I_RC]->K.transpose().inverse() * mE * mpCamera[I_LC]->K.inverse();
    mF /= mF(2, 2);
}

Eigen::Vector3d StereoCamera::Triangulate(const Eigen::Vector3d& ray_ll,
                                          const Eigen::Vector3d& ray_rr) {
    Eigen::Vector3d x3Dl;
    Eigen::Vector3d Al = ray_ll;
    Eigen::Vector3d Br = ray_rr;
    Eigen::Vector3d Bl = mTij[I_LC][I_RC].rotationMatrix() * Br + mTij[I_LC][I_RC].translation();

    Eigen::Vector3d AB = Bl-Al;
    Eigen::Vector3d d = ray_ll.normalized();
    Eigen::Vector3d l = (mTij[I_LC][I_RC].rotationMatrix() * ray_rr).normalized();

    // cramer rule
    double ltd = l.dot(d);
    double ltAB = l.dot(AB);
    double dtAB = d.dot(AB);
    double del = ltd * ltd - 1;

    if(std::abs(del) < 1e-8) {
        x3Dl << 0, 0, -1;
    }
    else {
        double inv_del = 1.0f / del;
        double delx = ltd * ltAB - dtAB;
        double dely = ltAB - ltd * dtAB;
        double x = delx * inv_del;
        double y = dely * inv_del;

        Eigen::Vector3d Cl = Al + x * d;
        Eigen::Vector3d Dl = Bl + y * l;
        x3Dl = (Cl + Dl) / 2;
    }
    return x3Dl;
}
