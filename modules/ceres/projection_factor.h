#pragma once
#include <ceres/ceres.h>
#include "camera_model/simple_stereo_camera.h"

class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const {
        Eigen::Map<const Sophus::SE3d> Tcw(parameters_raw[0]);
        Eigen::Map<const Eigen::Matrix<double, 3, 1>> x3Dw(parameters_raw[1]);

        Eigen::Map<Eigen::Vector2d> residual(residual_raw);
        Eigen::Vector3d x3Dc = Tcw * x3Dw;

        Eigen::Vector2d uv;
        mpCamera->Project2(x3Dc, uv);
        residual = pt - uv;

        if(jacobian_raw) {
            Eigen::Matrix<double, 2, 3> Jpi;
            if(jacobian_raw[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpose(jacobian_raw[0]);

            }

            if(jacobian_raw[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jpoint(jacobian_raw[1]);
            }
        }
    }

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector2d pt;
};

class StereoProjectionFactor : public ceres::SizedCostFunction<3, 7, 3>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    Eigen::Vector3d pt; // u v ur
};
