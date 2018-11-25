#include "projection_factor.h"

ProjectionFactor::ProjectionFactor(const SimpleStereoCamPtr& camera, const Eigen::Vector2d& pt_)
    : mpCamera(camera), pt(pt_) {}

bool ProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Tcw(parameters_raw[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> x3Dw(parameters_raw[1]);

    Eigen::Map<Eigen::Vector2d> residual(residual_raw);
    Eigen::Vector3d x3Dc = Tcw * x3Dw;

    Eigen::Vector2d uv;
    mpCamera->Project2(x3Dc, uv);
    residual = pt - uv;

    if(jacobian_raw) {
        Eigen::Matrix<double, 2, 3> Jpi_Xc;
        Jpi_Xc = mpCamera->J2(x3Dc);
        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> JXc_pose;
            JXc_pose << Eigen::Matrix3d::Identity(), -Sophus::SO3d::hat(x3Dc);
            Jpose.leftCols(6) = -Jpi_Xc * JXc_pose;
            Jpose.rightCols(1).setZero();
        }

        if(jacobian_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 3, Eigen::RowMajor>> Jpoint(jacobian_raw[1]);
            Jpoint = -Jpi_Xc;
        }
    }

    return true;
}

StereoProjectionFactor::StereoProjectionFactor(const SimpleStereoCamPtr& camera, const Eigen::Vector3d& pt_)
    : mpCamera(camera), pt(pt_) {}

bool StereoProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Tcw(parameters_raw[0]);
    Eigen::Map<const Eigen::Matrix<double, 3, 1>> x3Dw(parameters_raw[1]);

    Eigen::Map<Eigen::Vector3d> residual(residual_raw);
    Eigen::Vector3d x3Dc = Tcw * x3Dw;

    Eigen::Vector3d uv_ur;
    mpCamera->Project3(x3Dc, uv_ur);
    residual = pt - uv_ur;

    if(jacobian_raw) {
        Eigen::Matrix3d Jpi_Xc;
        Jpi_Xc = mpCamera->J3(x3Dc);

        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jpose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> JXc_pose;
            JXc_pose << Eigen::Matrix3d::Identity(), -Sophus::SO3d::hat(x3Dc);
            Jpose.leftCols(6) = -Jpi_Xc * JXc_pose;
            Jpose.rightCols(1).setZero();
        }

        if(jacobian_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> Jpoint(jacobian_raw[1]);
            Jpoint = -Jpi_Xc;
        }
    }

    return true;
}
