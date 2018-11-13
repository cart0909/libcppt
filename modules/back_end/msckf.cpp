#if 0
#include "msckf.h"

MsckfSystem::MsckfSystem() {

}

MsckfSystem::~MsckfSystem() {

}

void MsckfSystem::ProcessImuData(const ImuData& imu_data) {
    // Remove the bias from the measured gyro and acceleration
    Eigen::Vector3d gyro = imu_data.gyro - mImuState.gyro_bias;
    Eigen::Vector3d acc = imu_data.accel - mImuState.acc_bias;
    double dtime = imu_data.timestamp - mImuState.timestamp;

    // Compute discrete transition and noise covariance matrix
    Eigen::Matrix<double, 21, 21> F = Eigen::Matrix<double, 21, 21>::Zero();
    Eigen::Matrix<double, 21, 12> G = Eigen::Matrix<double, 21, 12>::Zero();

    // SMSCKF page 7
    F.block<3, 3>(0, 0) = -Sophus::SO3d::hat(gyro);
    F.block<3, 3>(0, 3) = -Eigen::Matrix3d::Identity();
    F.block<3, 3>(6, 0) = -mImuState.qig.inverse().matrix() * Sophus::SO3d::hat(acc);
    F.block<3, 3>(6, 9) = -mImuState.qig.inverse().matrix();
    F.block<3, 3>(12, 6) = Eigen::Matrix3d::Identity();

    G.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    G.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    G.block<3, 3>(6, 6) = -mImuState.qig.inverse().matrix();
    G.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

    // Approximate matrix exponential to the 3rd order,
    // which can be considered to be accurate enough assuming
    // dtime is within 0.01s
    Eigen::Matrix<double, 21, 21> Fdt = F * dtime;
    Eigen::Matrix<double, 21, 21> Fdt_square = Fdt * Fdt;
    Eigen::Matrix<double, 21, 21> Fdt_cube = Fdt_square * Fdt;
    Eigen::Matrix<double, 21, 21> Phi = Eigen::Matrix<double, 21, 21>::Identity() + Fdt +
            0.5*Fdt_square + (1.0/6.0)*Fdt_cube;

    // Propogate the state using 4-th order Runge-Kutta
    PredictNewState(dtime, gyro, acc, mImuState);

    // this is confuse for me
    // Modify the transition matrix
//    Matrix3d R_kk_1 = quaternionToRotation(imu_state.orientation_null);
//    Phi.block<3, 3>(0, 0) =
//      quaternionToRotation(imu_state.orientation) * R_kk_1.transpose();

//    Vector3d u = R_kk_1 * IMUState::gravity;
//    RowVector3d s = (u.transpose()*u).inverse() * u.transpose();

//    Matrix3d A1 = Phi.block<3, 3>(6, 0);
//    Vector3d w1 = skewSymmetric(
//        imu_state.velocity_null-imu_state.velocity) * IMUState::gravity;
//    Phi.block<3, 3>(6, 0) = A1 - (A1*u-w1)*s;

//    Matrix3d A2 = Phi.block<3, 3>(12, 0);
//    Vector3d w2 = skewSymmetric(
//        dtime*imu_state.velocity_null+imu_state.position_null-
//        imu_state.position) * IMUState::gravity;
//    Phi.block<3, 3>(12, 0) = A2 - (A2*u-w2)*s;
    // end
}

void MsckfSystem::PredictNewState(double dt, const Eigen::Vector3d& gyro,
                                  const Eigen::Vector3d& acc, ImuState& imu_state)
{
    Eigen::Vector3d gyrodt = gyro*dt;
    Eigen::Vector3d gyrodt_2 = gyrodt / 2;

    Sophus::SO3d Rigdt = Sophus::SO3d::exp(gyrodt) * imu_state.qig;
    Sophus::SO3d Rigdt_2 = Sophus::SO3d::exp(gyrodt_2) * imu_state.qig;
    Sophus::SO3d Rigdt_trans = Rigdt.inverse();
    Sophus::SO3d Rigdt_2_trans = Rigdt_2.inverse();

    // k1 = f(tn, yn)
    Eigen::Vector3d k1_v_dot = imu_state.qig.inverse().matrix()*acc + imu_state.g_gravity;
    Eigen::Vector3d k1_p_dot = imu_state.vgi;

    // k2 = f(tn+dt/2, yn+k1*dt/2)
    Eigen::Vector3d k1_v = imu_state.vgi + k1_v_dot * dt/2;
    Eigen::Vector3d k2_v_dot = Rigdt_2_trans.matrix() * acc + imu_state.g_gravity;
    Eigen::Vector3d k2_p_dot = k1_v;

    // K3 = f(tn+dt/2, yn+k2*dt/2)
    Eigen::Vector3d k2_v = imu_state.vgi + k2_v_dot * dt/2;
    Eigen::Vector3d k3_v_dot = Rigdt_2_trans.matrix() * acc + imu_state.g_gravity;
    Eigen::Vector3d k3_p_dot = k2_v;

    // K4 = f(tn+dt, yn+k3*dt)
    Eigen::Vector3d k3_v = imu_state.vgi + k3_v_dot * dt;
    Eigen::Vector3d k4_v_dot = Rigdt_trans.matrix() * acc + imu_state.g_gravity;
    Eigen::Vector3d k4_p_dot = k3_v;

    // yn+1 = yn + dt/6*(k1+2*k2+2*k3+k4)
    imu_state.qig = Rigdt;
    imu_state.vgi = imu_state.vgi + dt/6*(k1_v_dot + 2 * k2_v_dot + 2 * k3_v_dot + k4_v_dot);
    imu_state.pgi = imu_state.pgi + dt/6*(k1_p_dot + 2 * k2_p_dot + 2 * k3_p_dot + k4_p_dot);
}
#endif
