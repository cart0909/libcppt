#pragma once
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "basic_datatype/basic_sensor.h"

class CameraBase : public SensorBase
{
public:
    enum ModelType {
        PINHOLE,
        FISHEYE
    };

    CameraBase();
    CameraBase(ModelType model_type, const std::string& camera_name, int num_intrinsics,
               int image_width, int image_height);
    virtual ~CameraBase();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p) const = 0;
    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                 Eigen::Matrix<double, 2, 3>& J) const = 0;
    virtual void BackProject(const Eigen::Vector2d& p , Eigen::Vector3d& P) const = 0;

    ModelType mModelType;
    std::string mCameraName;
    int mnIntrinsics;
    int mnImageWidth;
    int mnImageHeight;
    cv::Mat mMask;
};

using CameraBasePtr = std::shared_ptr<CameraBase>;
using CameraBaseConstPtr = std::shared_ptr<const CameraBase>;
