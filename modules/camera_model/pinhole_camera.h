#pragma once
#include "camera.h"

class PinholeCamera : public CameraBase {
public:
};

using PinholeCameraPtr = std::shared_ptr<PinholeCamera>;
using PinholeCameraConstPtr = std::shared_ptr<const PinholeCamera>;
