#pragma once
#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <vector>

class ImuPreintegration {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuPreintegration();
    ~ImuPreintegration();

    void push_back(double dt, const Eigen::Vector3d& meas_gyr,
                   const Eigen::Vector3d& meas_acc);

    std::vector<double> mvdt;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > mvMeasGyr;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d> > mvMeasAcc;

    Eigen::Matrix<double, 6, 6> mGyrAccCov; // gyro and acc covariance Sigma_eta
    Eigen::Vector3d mbg, mba;

    Sophus::SO3d mdelRij;
    Eigen::Vector3d mdelVij;
    Eigen::Vector3d mdelPij;

    Eigen::Matrix3d mJR_bg;
    Eigen::Matrix3d mJV_ba;
    Eigen::Matrix3d mJV_bg;
    Eigen::Matrix3d mJP_ba;
    Eigen::Matrix3d mJP_bg;

    Eigen::Matrix<double, 9, 9> mCovariance; // phi, v, p

    double mdel_tij;
};
