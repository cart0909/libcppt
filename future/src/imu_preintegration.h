#pragma once
#include <Eigen/Dense>
#include <sophus/so3.hpp>
#include <vector>
#include "util.h"

class ImuPreintegration {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ImuPreintegration(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                      const Eigen::Matrix<double, 6, 6>& noise_cov);
    ~ImuPreintegration();

    void Clear();
    void push_back(double dt, const Eigen::Vector3d& meas_gyr,
                   const Eigen::Vector3d& meas_acc);
    void Propagate(double dt, const Eigen::Vector3d& meas_gyr,
                   const Eigen::Vector3d& meas_acc);
    void Repropagate(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba);

    std::vector<double> mvdt;
    Eigen::VecVector3d mvMeasGyr;
    Eigen::VecVector3d mvMeasAcc;

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
private:
    void Init();
};
SMART_PTR(ImuPreintegration)
