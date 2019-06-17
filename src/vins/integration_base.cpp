#include "integration_base.h"

IntegrationBase::IntegrationBase(const Eigen::Vector3d &_acc_0, const Eigen::Vector3d &_gyr_0,
                const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg,
                double acc_n, double gyr_n, double acc_w, double gyr_w)
    : acc_0{_acc_0}, gyr_0{_gyr_0}, linearized_acc{_acc_0}, linearized_gyr{_gyr_0},
      linearized_ba{_linearized_ba}, linearized_bg{_linearized_bg},
      jacobian{Eigen::Matrix15d::Identity()}, covariance{Eigen::Matrix15d::Zero()},
      sum_dt{0.0}, delta_p{Eigen::Vector3d::Zero()}, delta_q{Eigen::Quaterniond::Identity()}, delta_v{Eigen::Vector3d::Zero()}

{
    Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
    noise.setZero();
    noise.block<3, 3>(0, 0)   = (acc_n * acc_n) * I3x3;
    noise.block<3, 3>(3, 3)   = (gyr_n * gyr_n) * I3x3;
    noise.block<3, 3>(6, 6)   = (acc_n * acc_n) * I3x3;
    noise.block<3, 3>(9, 9)   = (gyr_n * gyr_n) * I3x3;
    noise.block<3, 3>(12, 12) = (acc_w * acc_w) * I3x3;
    noise.block<3, 3>(15, 15) = (gyr_w * gyr_w) * I3x3;
}

void IntegrationBase::push_back(double dt, const Eigen::Vector3d &acc, const Eigen::Vector3d &gyr)
{
    dt_buf.push_back(dt);
    acc_buf.push_back(acc);
    gyr_buf.push_back(gyr);
    propagate(dt, acc, gyr);
}

void IntegrationBase::repropagate(const Eigen::Vector3d &_linearized_ba, const Eigen::Vector3d &_linearized_bg)
{
    sum_dt = 0.0;
    acc_0 = linearized_acc;
    gyr_0 = linearized_gyr;
    delta_p.setZero();
    delta_q.setQuaternion(Eigen::Quaterniond(1, 0, 0, 0));
    delta_v.setZero();
    linearized_ba = _linearized_ba;
    linearized_bg = _linearized_bg;
    jacobian.setIdentity();
    covariance.setZero();
    for (int i = 0; i < static_cast<int>(dt_buf.size()); i++)
        propagate(dt_buf[i], acc_buf[i], gyr_buf[i]);
}

void IntegrationBase::midPointIntegration(double _dt,
                        const Eigen::Vector3d& _acc_0, const Eigen::Vector3d& _gyr_0,
                        const Eigen::Vector3d& _acc_1, const Eigen::Vector3d& _gyr_1,
                        const Eigen::Vector3d& delta_p, const Sophus::SO3d& delta_q, const Eigen::Vector3d& delta_v,
                        const Eigen::Vector3d& linearized_ba, const Eigen::Vector3d& linearized_bg,
                        Eigen::Vector3d& result_delta_p, Sophus::SO3d& result_delta_q, Eigen::Vector3d& result_delta_v,
                        Eigen::Vector3d& result_linearized_ba, Eigen::Vector3d& result_linearized_bg, bool update_jacobian)
{
    //ROS_INFO("midpoint integration");
    Eigen::Vector3d un_acc_0 = delta_q * (_acc_0 - linearized_ba);
    Eigen::Vector3d un_gyr = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
    result_delta_q = delta_q * Sophus::SO3d(Eigen::Quaterniond(1, un_gyr(0) * _dt / 2, un_gyr(1) * _dt / 2, un_gyr(2) * _dt / 2));
    Eigen::Vector3d un_acc_1 = result_delta_q * (_acc_1 - linearized_ba);
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    result_delta_p = delta_p + delta_v * _dt + 0.5 * un_acc * _dt * _dt;
    result_delta_v = delta_v + un_acc * _dt;
    result_linearized_ba = linearized_ba;
    result_linearized_bg = linearized_bg;

    if(update_jacobian)
    {
        Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
        Eigen::Vector3d w_x = 0.5 * (_gyr_0 + _gyr_1) - linearized_bg;
        Eigen::Vector3d a_0_x = _acc_0 - linearized_ba;
        Eigen::Vector3d a_1_x = _acc_1 - linearized_ba;
        Eigen::Matrix3d R_w_x, R_a_0_x, R_a_1_x;

        R_w_x<<0, -w_x(2), w_x(1),
            w_x(2), 0, -w_x(0),
            -w_x(1), w_x(0), 0;
        R_a_0_x<<0, -a_0_x(2), a_0_x(1),
            a_0_x(2), 0, -a_0_x(0),
            -a_0_x(1), a_0_x(0), 0;
        R_a_1_x<<0, -a_1_x(2), a_1_x(1),
            a_1_x(2), 0, -a_1_x(0),
            -a_1_x(1), a_1_x(0), 0;

        Eigen::Matrix15d F = Eigen::Matrix15d::Zero(15, 15);
        Eigen::Matrix3d deltaR = delta_q.matrix(), result_delta_R = result_delta_q.matrix();
        F.block<3, 3>(0, 0) = I3x3;
        F.block<3, 3>(0, 3) = -0.25 * deltaR * R_a_0_x * _dt * _dt +
                              -0.25 * result_delta_R * R_a_1_x * (I3x3 - R_w_x * _dt) * _dt * _dt;
        F.block<3, 3>(0, 6) = I3x3 * _dt;
        F.block<3, 3>(0, 9) = -0.25 * (deltaR + result_delta_R) * _dt * _dt;
        F.block<3, 3>(0, 12) = -0.25 * result_delta_R * R_a_1_x * _dt * _dt * -_dt;
        F.block<3, 3>(3, 3) = I3x3 - R_w_x * _dt;
        F.block<3, 3>(3, 12) = -1.0 * I3x3 * _dt;
        F.block<3, 3>(6, 3) = -0.5 * deltaR * R_a_0_x * _dt +
                              -0.5 * result_delta_R * R_a_1_x * (I3x3 - R_w_x * _dt) * _dt;
        F.block<3, 3>(6, 6) = I3x3;
        F.block<3, 3>(6, 9) = -0.5 * (deltaR + result_delta_R) * _dt;
        F.block<3, 3>(6, 12) = -0.5 * result_delta_R * R_a_1_x * _dt * -_dt;
        F.block<3, 3>(9, 9) = I3x3;
        F.block<3, 3>(12, 12) = I3x3;
        //cout<<"A"<<endl<<A<<endl;

        Eigen::Matrix15_18d V = Eigen::Matrix15_18d::Zero(15,18);
        V.block<3, 3>(0, 0) =  0.25 * deltaR * _dt * _dt;
        V.block<3, 3>(0, 3) =  0.25 * -result_delta_R * R_a_1_x  * _dt * _dt * 0.5 * _dt;
        V.block<3, 3>(0, 6) =  0.25 * result_delta_R * _dt * _dt;
        V.block<3, 3>(0, 9) =  V.block<3, 3>(0, 3);
        V.block<3, 3>(3, 3) =  0.5 * I3x3 * _dt;
        V.block<3, 3>(3, 9) =  0.5 * I3x3 * _dt;
        V.block<3, 3>(6, 0) =  0.5 * deltaR * _dt;
        V.block<3, 3>(6, 3) =  0.5 * -result_delta_R * R_a_1_x  * _dt * 0.5 * _dt;
        V.block<3, 3>(6, 6) =  0.5 * result_delta_R * _dt;
        V.block<3, 3>(6, 9) =  V.block<3, 3>(6, 3);
        V.block<3, 3>(9, 12) = I3x3 * _dt;
        V.block<3, 3>(12, 15) = I3x3 * _dt;

        jacobian = F * jacobian;
        covariance = F * covariance * F.transpose() + V * noise * V.transpose();
    }

}

void IntegrationBase::propagate(double _dt, const Eigen::Vector3d &_acc_1, const Eigen::Vector3d &_gyr_1)
{
    dt = _dt;
    acc_1 = _acc_1;
    gyr_1 = _gyr_1;
    Eigen::Vector3d result_delta_p;
    Sophus::SO3d result_delta_q;
    Eigen::Vector3d result_delta_v;
    Eigen::Vector3d result_linearized_ba;
    Eigen::Vector3d result_linearized_bg;

    midPointIntegration(_dt, acc_0, gyr_0, _acc_1, _gyr_1, delta_p, delta_q, delta_v,
                        linearized_ba, linearized_bg,
                        result_delta_p, result_delta_q, result_delta_v,
                        result_linearized_ba, result_linearized_bg, 1);

    //checkJacobian(_dt, acc_0, gyr_0, acc_1, gyr_1, delta_p, delta_q, delta_v,
    //                    linearized_ba, linearized_bg);
    delta_p = result_delta_p;
    delta_q = result_delta_q;
    delta_v = result_delta_v;
    linearized_ba = result_linearized_ba;
    linearized_bg = result_linearized_bg;
    delta_q.normalize();
    sum_dt += dt;
    acc_0 = acc_1;
    gyr_0 = gyr_1;
//    std::cout << "propagate sum_dt=" << sum_dt <<std::endl;
//   std::cout << "delta_p=" << delta_p <<std::endl;
//     std::cout << "acc_0=" << acc_0 <<std::endl;

}

Eigen::Vector15d IntegrationBase::evaluate(const Eigen::Vector3d& Pi, const Sophus::SO3d& Qi, const Eigen::Vector3d& Vi, const Eigen::Vector3d& Bai, const Eigen::Vector3d& Bgi,
                                           const Eigen::Vector3d& Pj, const Sophus::SO3d& Qj, const Eigen::Vector3d& Vj, const Eigen::Vector3d& Baj, const Eigen::Vector3d& Bgj,
                                           const Eigen::Vector3d& Gw)
{
    Eigen::Vector15d residuals;

    Eigen::Matrix3d dp_dba = jacobian.block<3, 3>(O_P, O_BA);
    Eigen::Matrix3d dp_dbg = jacobian.block<3, 3>(O_P, O_BG);

    Eigen::Matrix3d dq_dbg = jacobian.block<3, 3>(O_R, O_BG);

    Eigen::Matrix3d dv_dba = jacobian.block<3, 3>(O_V, O_BA);
    Eigen::Matrix3d dv_dbg = jacobian.block<3, 3>(O_V, O_BG);

    Eigen::Vector3d dba = Bai - linearized_ba;
    Eigen::Vector3d dbg = Bgi - linearized_bg;

    Sophus::SO3d corrected_delta_q = delta_q * Sophus::SO3d::exp(dq_dbg * dbg);
    Eigen::Vector3d corrected_delta_v = delta_v + dv_dba * dba + dv_dbg * dbg;
    Eigen::Vector3d corrected_delta_p = delta_p + dp_dba * dba + dp_dbg * dbg;

    residuals.block<3, 1>(O_P, 0) = Qi.inverse() * (0.5 * Gw * sum_dt * sum_dt + Pj - Pi - Vi * sum_dt) - corrected_delta_p;
    residuals.block<3, 1>(O_R, 0) = (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).log();
    residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (Gw * sum_dt + Vj - Vi) - corrected_delta_v;
    residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
    residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;
    return residuals;
}
