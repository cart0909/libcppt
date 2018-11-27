#pragma once
#include <memory>
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>

using ImagePyr = std::vector<cv::Mat>;
using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
using VecVector2f = std::vector<Eigen::Vector2f, Eigen::aligned_allocator<Eigen::Vector2f>>;
using VecVector3f = std::vector<Eigen::Vector3f, Eigen::aligned_allocator<Eigen::Vector3f>>;

#define SMART_PTR(NAME) \
using NAME##Ptr = std::shared_ptr<NAME>; \
using NAME##ConstPtr = std::shared_ptr<const NAME>; \
using NAME##WPtr = std::weak_ptr<NAME>; \
using NAME##ConstWPtr = std::weak_ptr<const NAME>;

template<class T>
void ReduceVector(std::vector<T> &v, const std::vector<uchar>& status) {
    int j = 0;
    for(int i = 0, n = v.size(); i < n; ++i) {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

bool InBorder(const cv::Point2f& pt, int width, int height);
