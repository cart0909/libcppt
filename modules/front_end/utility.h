#pragma once
#include <cassert>
#include <opencv2/opencv.hpp>

using ImagePyr = std::vector<cv::Mat>;
namespace Utility {
ImagePyr Pyramid(const cv::Mat& img, int num_levels);
}
