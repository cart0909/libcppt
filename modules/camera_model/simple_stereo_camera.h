#pragma once
#include <memory>
#include <opencv2/opencv.hpp>
#include "basic_datatype/basic_sensor.h"

class SimpleStereoCam : public SensorBase {
    SimpleStereoCam() = delete;
    SimpleStereoCam(const Sophus::SE3d& Tbs_, int width_, int height_,
                    double f_, double cx_, double cy_, double b_,
                    const cv::Mat& M1l_, const cv::Mat& M2l_,
                    const cv::Mat& M1r_, const cv::Mat& M2r_);
    virtual ~SimpleStereoCam();

    // left cam api
    void Project2(const Eigen::Vector3d& P, Eigen::Vector2d& p) const;
    void Project2(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                  Eigen::Matrix<double, 2, 3>& J) const;
    // P(2) = 1
    void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;

    // stereo cam api
    // p is [u, v, ur]'
    void Project3(const Eigen::Vector3d& P, Eigen::Vector3d& p) const;
    // p is [u, v, ur]'
    void Project3(const Eigen::Vector3d& P, Eigen::Vector3d& p,
                  Eigen::Matrix3d& J) const;
    // p is [u, v, ur]'
    void Triangulate(const Eigen::Vector3d& p, Eigen::Vector3d& P) const;


    const int width;
    const int height;
    const double f;
    const double inv_f;
    const double cx;
    const double cy;
    const double b;
    const double bf;

    // rectify table
    cv::Mat M1l, M2l;
    cv::Mat M1r, M2r;
};

using SimpleStereoCamPtr = std::shared_ptr<SimpleStereoCam>;
using SimpleStereoCamConstPtr = std::shared_ptr<const SimpleStereoCam>;
