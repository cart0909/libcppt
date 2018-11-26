#pragma once
#include <ceres/ceres.h>
#include "camera_model/simple_stereo_camera.h"

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionFactor(const SimpleStereoCamPtr& camera, const Eigen::Vector2d& pt_);

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector2d pt;
};

class StereoProjectionFactor : public ceres::SizedCostFunction<3, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StereoProjectionFactor(const SimpleStereoCamPtr& camera, const Eigen::Vector3d& pt_);

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector3d pt; // u v ur
};

namespace unary {

class ProjectionFactor : public ceres::SizedCostFunction<2, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

class StereoProjectionFactor : public ceres::SizedCostFunction<3, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

};
