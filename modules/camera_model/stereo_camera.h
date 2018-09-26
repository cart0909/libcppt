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
};

using StereoCameraPtr = std::shared_ptr<StereoCamera>;
using StereoCameraConstPtr = std::shared_ptr<const StereoCamera>;
