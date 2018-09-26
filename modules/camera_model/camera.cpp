#include "camera.h"

CameraBase::CameraBase() {}

CameraBase::CameraBase(ModelType model_type, const std::string& camera_name, int num_intrinsics,
                       int image_width, int image_height)
    : mModelType(model_type), mCameraName(camera_name), mnIntrinsics(num_intrinsics), mnImageWidth(image_width),
      mnImageHeight(image_height)
{
    mMask = cv::Mat(image_height, image_width, CV_8U, cv::Scalar(255));
}

CameraBase::~CameraBase() {}
