#include "simple_stereo_camera.h"

SimpleStereoCam::SimpleStereoCam(const Sophus::SE3d& Tbs_, int width_, int height_,
                                 double f_, double cx_, double cy_, double b_,
                                 const cv::Mat& M1l_, const cv::Mat& M2l_,
                                 const cv::Mat& M1r_, const cv::Mat& M2r_)
    : SensorBase(Tbs_), width(width_), height(height_), f(f_), inv_f(1.0f/f),
      cx(cx_), cy(cy_), b(b_), bf(b*f),
      M1l(M1l_.clone()), M2l(M2l_.clone()), M1r(M1r_.clone()), M2r(M2r_.clone())
{

}

SimpleStereoCam::~SimpleStereoCam() {}

void SimpleStereoCam::Project2(const Eigen::Vector3d& P, Eigen::Vector2d& p) const {
    double inv_z = 1.0f / P(2);
    double f_inv_z = f * inv_z;
    p(0) = P(0) * f_inv_z + cx;
    p(1) = P(1) * f_inv_z + cy;
}

void SimpleStereoCam::Project2(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                               Eigen::Matrix<double, 2, 3>& J) const {
    double inv_z = 1.0f / P(2);
    double f_inv_z = f * inv_z;
    double f_inv_z2 = f_inv_z * inv_z;
    p(0) = P(0) * f_inv_z + cx;
    p(1) = P(1) * f_inv_z + cy;

    J << f_inv_z, 0, -f_inv_z2 * P(0),
         0, f_inv_z, -f_inv_z2 * P(1);
}

Eigen::Matrix<double, 2, 3> SimpleStereoCam::J2(const Eigen::Vector3d& P) const {
    Eigen::Matrix<double, 2, 3> J;
    double inv_z = 1.0f / P(2);
    double f_inv_z = f * inv_z;
    double f_inv_z2 = f_inv_z * inv_z;

    J << f_inv_z, 0, -f_inv_z2 * P(0),
         0, f_inv_z, -f_inv_z2 * P(1);
    return J;
}

// P(2) = 1
void SimpleStereoCam::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {
    P << (p(0) - cx) * inv_f, (p(1) - cy) * inv_f, 1;
}

void SimpleStereoCam::Project3(const Eigen::Vector3d& P, Eigen::Vector3d& p) const {
    double inv_z = 1.0f / P(2);
    double f_inv_z = f * inv_z;
    p(0) = P(0) * f_inv_z + cx; // u
    p(1) = P(1) * f_inv_z + cy; // v
    p(2) = p(0) - bf * inv_z;   // ur
}

void SimpleStereoCam::Project3(const Eigen::Vector3d& P, Eigen::Vector3d& p,
                               Eigen::Matrix3d& J) const {
    double inv_z = 1.0f / P(2);
    double inv_z2 = inv_z * inv_z;
    double f_inv_z = f * inv_z;
    double f_inv_z2 = f * inv_z2;

    p(0) = P(0) * f_inv_z + cx; // u
    p(1) = P(1) * f_inv_z + cy; // v
    p(2) = p(0) - bf * inv_z;   // ur

    J << f_inv_z, 0, -f_inv_z2 * P(0),
         0, f_inv_z, -f_inv_z2 * P(1),
         f_inv_z, 0, (bf - f * P(0)) * inv_z2;
}

Eigen::Matrix3d SimpleStereoCam::J3(const Eigen::Vector3d& P) const {
    Eigen::Matrix3d J;
    double inv_z = 1.0f / P(2);
    double inv_z2 = inv_z * inv_z;
    double f_inv_z = f * inv_z;
    double f_inv_z2 = f * inv_z2;

    J << f_inv_z, 0, -f_inv_z2 * P(0),
         0, f_inv_z, -f_inv_z2 * P(1),
         f_inv_z, 0, (bf - f * P(0)) * inv_z2;
    return J;
}

void SimpleStereoCam::Triangulate(const Eigen::Vector3d& p, Eigen::Vector3d& P) const {
    double disparity = p(0) - p(2); // ul - ur
    double z = bf / disparity;
    P << (p(0) - cx) * inv_f, (p(1) - cy) * inv_f, 1;
    P *= z;
}
