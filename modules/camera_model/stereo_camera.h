#include "basic_datatype/basic_sensor.h"
#include "pinhole_camera.h"

class StereoCamera {
public:
    enum SensorIdx {
      I_LC,
      I_RC,
      I_IMU,
      NUM_OF_SENSOR
    };

    StereoCamera(const ImuSensorPtr& imu_sensor,
                 const PinholeCameraPtr& left_cam,
                 const PinholeCameraPtr& right_cam);

    ImuSensorPtr mpImuSensor;
    PinholeCameraPtr mpCamera[2];
    Sophus::SE3d mTij[NUM_OF_SENSOR][NUM_OF_SENSOR];
    bool mbNoImuSensor;

    Eigen::Matrix3d mE; // essential matrix yl' * E * yr = 0
    Eigen::Matrix3d mF; // fundamental matrix xl' * F * xr = 0
};

using StereoCameraPtr = std::shared_ptr<StereoCamera>;
using StereoCameraConstPtr = std::shared_ptr<const StereoCamera>;
