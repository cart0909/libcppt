#include "imu_preintegration.h"
#include "basic_datatype/so3_extent.h"

ImuPreintegration::ImuPreintegration() {
    Init();
}

ImuPreintegration::ImuPreintegration(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba,
                    const Eigen::Matrix<double, 6, 6>& noise_cov) {
    Init();
    mbg = bg;
    mba = ba;
    mGyrAccCov = noise_cov;
}

ImuPreintegration::~ImuPreintegration() {

}

void ImuPreintegration::Clear() {
    mvdt.clear();
    mvMeasGyr.clear();
    mvMeasAcc.clear();
    Init();
}

void ImuPreintegration::Init() {
    mdelRij.setQuaternion(Eigen::Quaterniond(1, 0, 0, 0));
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

void ImuPreintegration::push_back(double dt, const Eigen::Vector3d& meas_gyr,
                                  const Eigen::Vector3d& meas_acc) {
    mvdt.emplace_back(dt);
    mvMeasGyr.emplace_back(meas_gyr);
    mvMeasAcc.emplace_back(meas_acc);
    Propagate(dt, meas_gyr, meas_acc);
}

void ImuPreintegration::Propagate(double dt, const Eigen::Vector3d& meas_gyr,
                                  const Eigen::Vector3d& meas_acc) {
    double dt2 = dt * dt;

    Eigen::Vector3d omega = meas_gyr - mbg;
    Eigen::Vector3d acc = meas_acc - mba;

    Sophus::SO3d dR = Sophus::SO3d::exp(omega*dt);
    Eigen::Matrix3d Jr = Sophus::JacobianR(omega*dt);

    Eigen::Matrix3d delRij_1 = mdelRij.matrix();
    Eigen::Matrix3d acc_hat = Sophus::SO3d::hat(acc);
    Eigen::Matrix3d delRij_1_acc_hat = delRij_1 * acc_hat;

    // noise covariance propagation of delta measurements
    // paper: Supplementary Material to: IMU Preintegration on Manifold for
    // Efficient Visual-Inertial Maximum-a-Posteriori Estimation (A.8) (A.9)
    Eigen::Matrix<double, 9, 9> A_phi_v_p = Eigen::Matrix<double, 9, 9>::Identity();
    Eigen::Matrix<double, 9, 6> B_gyr_acc = Eigen::Matrix<double, 9, 6>::Zero();
    A_phi_v_p.block<3, 3>(0, 0) = dR.inverse().matrix();
    A_phi_v_p.block<3, 3>(3, 0) = - delRij_1_acc_hat * dt;
    A_phi_v_p.block<3, 3>(6, 0) = - 0.5 * delRij_1_acc_hat * dt2;
    A_phi_v_p.block<3, 3>(6, 3) = Eigen::Matrix3d::Identity() * dt;

    B_gyr_acc.block<3, 3>(0, 0) = Jr * dt;
    B_gyr_acc.block<3, 3>(3, 3) = delRij_1 * dt;
    B_gyr_acc.block<3, 3>(6, 3) = 0.5 * delRij_1 * dt2;

    mCovariance = A_phi_v_p * mCovariance * A_phi_v_p.transpose() +
            B_gyr_acc * mGyrAccCov * B_gyr_acc.transpose();

    // jacobian of delta measurements w.r.t bias of gyro/acc
    // update P fitst, then V, then R
    mJP_ba += mJV_ba * dt - 0.5 * delRij_1 * dt2;
    mJP_bg += mJV_bg * dt - 0.5 * delRij_1_acc_hat * mJR_bg * dt2;
    mJV_ba += -delRij_1 * dt;
    mJV_bg += -delRij_1_acc_hat * mJR_bg * dt;
    mJR_bg = dR.inverse().matrix() * mJR_bg - Jr * dt;


    // compute delta measurements, position/velocity/rotation(matrix)
    // update P first, then V, then R. because P's update need V&R's
    // previous state
    mdelPij += mdelVij * dt + 0.5 * delRij_1 * acc * dt2;
    mdelVij += delRij_1 * acc * dt;
    mdelRij = mdelRij * dR;

    mdel_tij += dt;
}

void ImuPreintegration::Repropagate(const Eigen::Vector3d& bg, const Eigen::Vector3d& ba) {
    Init();
    mbg = bg;
    mba = ba;
    for(int i = 0, n = mvdt.size(); i < n; ++i) {
        Propagate(mvdt[i], mvMeasGyr[i], mvMeasAcc[i]);
    }
}
