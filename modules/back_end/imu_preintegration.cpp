#include "imu_preintegration.h"
#include "basic_datatype/so3_extent.h"

ImuPreintegration::ImuPreintegration() {
    mdelVij.setZero();
    mdelPij.setZero();

    mJR_bg.setZero();
    mJV_bg.setZero();
    mJV_ba.setZero();
    mJP_bg.setZero();
    mJP_ba.setZero();

    mCovariance.setZero();

    mdel_tij = 0.0f;
}

ImuPreintegration::~ImuPreintegration() {

}

void ImuPreintegration::push_back(double dt, const Eigen::Vector3d& meas_gyr,
                                  const Eigen::Vector3d& meas_acc) {
    double dt2 = dt * dt;

    mvdt.emplace_back(dt);
    mvMeasGyr.emplace_back(meas_gyr);
    mvMeasAcc.emplace_back(meas_acc);

    Eigen::Vector3d omega = meas_gyr - mbg;
    Eigen::Vector3d acc = meas_acc - mba;

    Sophus::SO3d dR = Sophus::SO3d::exp(omega*dt);
    Eigen::Matrix3d Jr = Sophus::JacobianR(omega*dt);

    // noise covariance propagation of delta measurements
    // paper: Supplementary Material to: IMU Preintegration on Manifold for
    // Efficient Visual-Inertial Maximum-a-Posteriori Estimation (A.8) (A.9)
    Eigen::Matrix<double, 9, 9> A_phi_v_p = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 6> B_gyr_acc = Eigen::Matrix<double, 9, 6>::Identity();

    // jacobian of delta measurements w.r.t bias of gyro/acc
    // update P fitst, then V, then R
    Eigen::Matrix3d delRij_1 = mdelRij.matrix();
    Eigen::Matrix3d acc_hat = Sophus::SO3d::hat(acc);
    mJP_ba += mJV_ba * dt - 0.5 * delRij_1 * dt2;
    mJP_bg += mJV_bg * dt - 0.5 * delRij_1 * acc_hat * mJR_bg * dt2;
    mJV_ba += -delRij_1 * dt;
    mJV_bg += -delRij_1 * acc_hat * mJR_bg * dt;
    mJR_bg = dR.inverse().matrix() * mJR_bg - Jr * dt;


    // compute delta measurements, position/velocity/rotation(matrix)
    // update P first, then V, then R. because P's update need V&R's
    // previous state
    mdelPij += mdelVij * dt + 0.5 * delRij_1 * acc * dt2;
    mdelVij += delRij_1 * acc * dt;
    mdelRij = mdelRij * dR;

    mdel_tij += dt;
}
