#pragma once
#include <memory>
#include <opencv2/opencv.hpp>

class CameraBase {
public:
    enum ModelType {
        PINHOLE,
        FISHEYE
    };

    CameraBase();
    virtual ~CameraBase();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p) = 0;
    virtual void BackProject(const Eigen::Vector2d& p , Eigen::Vector3d& P) = 0;

    ModelType mModelType;
    std::string mCameraName;
    int mnIntrinsics;
    int mnImageWidth;
    int mnImageHeight;
    cv::Mat mMask;
};

using CameraBasePtr = std::shared_ptr<CameraBase>;
using CameraBaseConstPtr = std::shared_ptr<const CameraBase>;
