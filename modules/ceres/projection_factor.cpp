#include "projection_factor.h"
#include <ros/ros.h>

ProjectionFactor::ProjectionFactor(const SimpleStereoCamPtr& camera,
                                   const Eigen::Vector2d& pt_j_,
                                   const Eigen::Vector2d& pt_i)
    : mpCamera(camera), pt_j(pt_j_)
{
    mpCamera->BackProject(pt_i, pt_i_normal_plane);
}

bool ProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twcj(parameters_raw[0]);
    double inv_zi = parameters_raw[1][0];
    Eigen::Map<const Sophus::SE3d> Twci(parameters_raw[2]);
    Eigen::Map<Eigen::Vector2d> residual(residual_raw);

    Sophus::SE3d Tcj_ci = Twcj.inverse() * Twci;
    Eigen::Vector3d x3Di = pt_i_normal_plane / inv_zi;
    Eigen::Vector3d x3Dj = Tcj_ci * x3Di;
    Eigen::Vector2d uv;
    mpCamera->Project2(x3Dj, uv);
    residual = uv - pt_j;

    if(jacobian_raw) {
        const Eigen::Matrix<double, 2, 3> Jpi_Xc = mpCamera->J2(x3Dj);

        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpose_j(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> JXc_pose_j;
            JXc_pose_j << -Twcj.so3().inverse().matrix(), Sophus::SO3d::hat(x3Dj);
            Jpose_j.leftCols<6>() = Jpi_Xc * JXc_pose_j;
            Jpose_j.rightCols<1>().setZero();
        }

        if(jacobian_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 1>> Jpoint(jacobian_raw[1]);
            Jpoint = Jpi_Xc * (-Tcj_ci.rotationMatrix() * pt_i_normal_plane / (inv_zi * inv_zi));
        }

        if(jacobian_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpose_i(jacobian_raw[2]);
            Eigen::Matrix<double, 3, 6> JXc_pose_i;
            JXc_pose_i << Twcj.so3().inverse().matrix(), -Tcj_ci.rotationMatrix() * Sophus::SO3d::hat(x3Di);
            Jpose_i.leftCols<6>() = Jpi_Xc * JXc_pose_i;
            Jpose_i.rightCols<1>().setZero();
        }
    }

    return true;
}

StereoProjectionFactor::StereoProjectionFactor(const SimpleStereoCamPtr& camera,
                                               const Eigen::Vector3d& pt_j_,
                                               const Eigen::Vector2d& pt_i)
    : mpCamera(camera), pt_j(pt_j_) {
    mpCamera->BackProject(pt_i, pt_i_normal_plane);
}

bool StereoProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twcj(parameters_raw[0]);
    double inv_zi = parameters_raw[1][0];
    Eigen::Map<const Sophus::SE3d> Twci(parameters_raw[2]);
    Eigen::Map<Eigen::Vector3d> residual(residual_raw);

    Sophus::SE3d Tcj_ci = Twcj.inverse() * Twci;
    Eigen::Vector3d x3Di = pt_i_normal_plane / inv_zi;
    Eigen::Vector3d x3Dj = Tcj_ci * x3Di;

    Eigen::Vector3d uv_ur;
    mpCamera->Project3(x3Dj, uv_ur);
    residual = uv_ur - pt_j;

    if(jacobian_raw) {
        Eigen::Matrix3d Jpi_Xc;
        Jpi_Xc = mpCamera->J3(x3Dj);

        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jpose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> JXc_pose;
            JXc_pose << -Twcj.so3().inverse().matrix(), Sophus::SO3d::hat(x3Dj);
            Jpose.leftCols<6>() = Jpi_Xc * JXc_pose;
            Jpose.rightCols<1>().setZero();
        }

        if(jacobian_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 3, 1>> Jpoint(jacobian_raw[1]);
            Jpoint = Jpi_Xc * (-Tcj_ci.rotationMatrix() * pt_i_normal_plane / (inv_zi * inv_zi));
        }

        if(jacobian_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jpose_i(jacobian_raw[2]);
            Eigen::Matrix<double, 3, 6> JXc_pose_i;
            JXc_pose_i << Twcj.so3().inverse().matrix(), -Tcj_ci.rotationMatrix() * Sophus::SO3d::hat(x3Di);
            Jpose_i.leftCols<6>() = Jpi_Xc * JXc_pose_i;
            Jpose_i.rightCols<1>().setZero();
        }
    }

    return true;
}

namespace unary {

ProjectionFactor::ProjectionFactor(const SimpleStereoCamPtr& camera_, const Eigen::Vector2d& pt_,
                 const Eigen::Vector3d& x3Dw_)
    : camera(camera_), pt(pt_), x3Dw(x3Dw_)
{}

bool ProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                                double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twc(parameters_raw[0]);
    Eigen::Map<Eigen::Vector2d> residual(residual_raw);

    Eigen::Vector3d x3Dc = Twc.inverse() * x3Dw;
    Eigen::Vector2d uv;
    camera->Project2(x3Dc, uv);
    residual = uv - pt;

    if(jacobian_raw) {
        Eigen::Matrix<double, 2, 3> Jpi_x3Dc;
        Jpi_x3Dc = camera->J2(x3Dc);
        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jpi_pose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> Jx3Dc_pose;
            Jx3Dc_pose << -Twc.so3().inverse().matrix(), Sophus::SO3d::hat(x3Dc);
            Jpi_pose.leftCols(6) = Jpi_x3Dc * Jx3Dc_pose;
            Jpi_pose.rightCols(1).setZero();
        }
    }
    return true;
}

StereoProjectionFactor::StereoProjectionFactor(const SimpleStereoCamPtr& camera_, const Eigen::Vector3d& pt_,
                                               const Eigen::Vector3d& x3Dw_)
    : camera(camera_), pt(pt_), x3Dw(x3Dw_)
{}

bool StereoProjectionFactor::Evaluate(double const* const* parameters_raw, double *residual_raw,
                                      double **jacobian_raw) const {
    Eigen::Map<const Sophus::SE3d> Twc(parameters_raw[0]);
    Eigen::Map<Eigen::Vector3d> residual(residual_raw);

    Eigen::Vector3d x3Dc = Twc.inverse() * x3Dw;
    Eigen::Vector3d uv_ur;
    camera->Project3(x3Dc, uv_ur);
    residual = uv_ur - pt;

    if(jacobian_raw) {
        Eigen::Matrix3d Jpi_x3Dc;
        Jpi_x3Dc = camera->J3(x3Dc);
        if(jacobian_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> Jpi_pose(jacobian_raw[0]);
            Eigen::Matrix<double, 3, 6> Jx3Dc_pose;
            Jx3Dc_pose << -Twc.so3().inverse().matrix(), Sophus::SO3d::hat(x3Dc);
            Jpi_pose.leftCols(6) = Jpi_x3Dc * Jx3Dc_pose;
            Jpi_pose.rightCols(1).setZero();
        }
    }
    return true;
}

}
