#pragma once
#include "ceres/local_parameterization_se3.h"
#include <opencv2/opencv.hpp>
#include <camera_model/pinhole_camera.h>

class IntensityFactor : public ceres::SizedCostFunction<16, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    IntensityFactor(PinholeCameraConstPtr camera, const cv::Mat& img_ref,
                    const cv::Mat& img_cur, const Eigen::Vector3d& _x3Dr,
                    const Eigen::Vector2d& uv_r,
                    int level);

    ~IntensityFactor() {}

    bool Init();
    virtual bool Evaluate(double const* const* parameters_raw,
                          double* residuals_raw, double** jacobian_raw) const;

    static const int patch_halfsize;
    static const int patch_size;
    static const int patch_area;

    int mLevel;
    Eigen::Vector3d x3Dr;
    Eigen::Vector2d uv_r;
    cv::Mat mImgRef, mImgCur;
    PinholeCameraConstPtr mpCamera;
    Eigen::Matrix<double, 16, 1> mRefPatch;
    Eigen::Matrix<double, 16, 6, Eigen::RowMajor> mJacobian;
};
