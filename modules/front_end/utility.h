#pragma once
#include <cassert>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include "basic_datatype/util_datatype.h"

namespace Utility {
ImagePyr Pyramid(const cv::Mat& img, int num_levels);
}
