#pragma once
#include "camera.h"

class PinholeCamera : public CameraBase {
public:
    PinholeCamera() = delete;
    PinholeCamera(const std::string& camera_name, int image_width, int image_height,
                  double fx_, double fy_, double cx_, double cy_,
                  double k1_ = 0.0, double k2_ = 0.0, double p1_ = 0.0, double p2_ = 0.0);
    ~PinholeCamera();

    void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p) const override;
    void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                 Eigen::Matrix<double, 2, 3>& J) const;
    void BackProject(const Eigen::Vector2d& p , Eigen::Vector3d& P) const override;

    void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const;
    void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,
                    Eigen::Matrix2d& J) const;

    double fx;
    double fy;
    double cx;
    double cy;
    double k1;
    double k2;
    double p1;
    double p2;
    bool mbNoDistortion;
    double inv_K00;
    double inv_K11;
    double inv_K02;
    double inv_K12;

    Eigen::Matrix3d K; // camera matrix
};

using PinholeCameraPtr = std::shared_ptr<PinholeCamera>;
using PinholeCameraConstPtr = std::shared_ptr<const PinholeCamera>;
