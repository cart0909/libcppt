#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "basic_datatype/imu_data.h"
#include "camera_model/stereo_camera.h"
#include "back_end/imu_state.h"
#include "back_end/cam_state.h"

class MsckfSystem {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MsckfSystem();
    ~MsckfSystem();

    void LoadParamFromYaml(const std::string& filename);
    void InitSystem();

    void EnqueueData();
    void ProcessImuData(const ImuData& imu_data);
    void PredictNewState(double dt, const Eigen::Vector3d& gyro,
                         const Eigen::Vector3d& acc, ImuState& imu_state);
    void HandleFeatureData();
    void InitGravityAndBias();

    StereoCameraPtr mpStereoCam;
    bool mbLoadParam;
    bool mbGravitySet;
    std::vector<ImuData> mvImuBuffer;

    ImuState mImuState;
};
