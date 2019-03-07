#include "projection_td_factor.h"

ProjectionTdFactor::ProjectionTdFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& velocity_i_, double td_i_,
                                       const Eigen::Vector3d& pt_j_, const Eigen::Vector3d& velocity_j_, double td_j_,
                                       double focal_length)
    : td_i(td_i_), td_j(td_j_), pt_i(pt_i_), pt_j(pt_j_), velocity_i(velocity_i_), velocity_j(velocity_j_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
}

bool ProjectionTdFactor::Evaluate(double const * const *parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    //parameters [0]: frame i
    //           [1]: frame j
    //           [2]: Tbc
    //           [3]: depth
    //           [4]: td
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Eigen::Map<const Sophus::SO3d> q_bc(parameters_raw[2]);
    Eigen::Map<const Eigen::Vector3d> p_bc(parameters_raw[2] + 4);
    double inv_zi = parameters_raw[3][0];
    double td = parameters_raw[4][0];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Vector3d pt_i_td, pt_j_td;
    // original formular pt_td = pt + td * velocity
    pt_i_td = pt_i - (td - td_i) * velocity_i; // t(pt_i) = img_t + td_i
    pt_j_td = pt_j - (td - td_j) * velocity_j; // td -> img_t + td

    Eigen::Vector3d x3Dci = pt_i_td / inv_zi;
    Eigen::Vector3d x3Dbi = q_bc * x3Dci + p_bc;
    Eigen::Vector3d x3Dw = q_wbi * x3Dbi + p_wbi;
    Eigen::Vector3d x3Dbj = q_wbj.inverse() * (x3Dw - p_wbj);
    Eigen::Vector3d x3Dcj = q_bc.inverse() * (x3Dbj - p_bc);
    double zj = x3Dcj(2);
    residuals = (x3Dcj / zj).head<2>() - pt_j_td.head<2>();
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce;
        double inv_zj = 1.0f / zj;
        double inv_zj2 = inv_zj * inv_zj;
        reduce << inv_zj, 0, -x3Dcj(0) * inv_zj2,
                  0, inv_zj, -x3Dcj(1) * inv_zj2;

        reduce = sqrt_info * reduce;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = (q_bc.inverse() * q_wbj.inverse()).matrix();
            J.rightCols<3>() = (q_bc.inverse() * q_wbj.inverse() * q_wbi).matrix() * -Sophus::SO3d::hat(x3Dbi);
            J_pose_i.leftCols<6>() = reduce * J;
            J_pose_i.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[1]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = -(q_wbj * q_bc).inverse().matrix();
            J.rightCols<3>() = q_bc.inverse().matrix() * Sophus::SO3d::hat(x3Dbj);
            J_pose_j.leftCols<6>() = reduce * J;
            J_pose_j.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_bc(jacobians_raw[2]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = q_bc.inverse().matrix() * ((q_wbj.inverse() * q_wbi).matrix() - Eigen::Matrix3d::Identity());
            J.rightCols<3>() = Sophus::SO3d::hat(x3Dcj) - (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc).matrix() * Sophus::SO3d::hat(x3Dci);
            J_bc.leftCols<6>() = reduce * J;
            J_bc.rightCols<1>().setZero();
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Vector2d> J_feat(jacobians_raw[3]);
            J_feat = reduce * (-1.0 / (inv_zi * inv_zi)) *
                    (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc * pt_i_td);
        }

        if(jacobians_raw[4]) {
            Eigen::Map<Eigen::Vector2d> J_td(jacobians_raw[4]);
            J_td = sqrt_info * velocity_j.head<2>() - reduce * (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc * velocity_i / inv_zi);
        }
    }
    return true;
}

SlaveProjectionTdFactor::SlaveProjectionTdFactor(const Eigen::Vector3d& pt_mi_, const Eigen::Vector3d& velocity_i_, double td_i_,
                                                 const Eigen::Vector3d& pt_sj_, const Eigen::Vector3d& velocity_j_, double td_j_,
                                                 double focal_length)
    : td_i(td_i_), td_j(td_j_), pt_mi(pt_mi_), pt_sj(pt_sj_), velocity_i(velocity_i_), velocity_j(velocity_j_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
}

bool SlaveProjectionTdFactor::Evaluate(double const * const *parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    //parameters [0]: frame i
    //           [1]: frame j
    //           [2]: Tbc
    //           [3]: Tsm
    //           [4]: depth
    //           [5]: td
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Eigen::Map<const Sophus::SO3d> q_bc(parameters_raw[2]);
    Eigen::Map<const Eigen::Vector3d> p_bc(parameters_raw[2] + 4);
    Eigen::Map<const Sophus::SO3d> q_sm(parameters_raw[3]);
    Eigen::Map<const Eigen::Vector3d> p_sm(parameters_raw[3] + 4);
    double inv_zi = parameters_raw[4][0];
    double td = parameters_raw[5][0];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Vector3d pt_mi_td, pt_sj_td;
    // original formular pt_td = pt + td * velocity
    pt_mi_td = pt_mi - (td - td_i) * velocity_i;
    pt_sj_td = pt_sj - (td - td_j) * velocity_j;

    Eigen::Vector3d x3Dmi = pt_mi_td / inv_zi;
    Eigen::Vector3d x3Dbi = q_bc * x3Dmi + p_bc;
    Eigen::Vector3d x3Dw = q_wbi * x3Dbi + p_wbi;
    Eigen::Vector3d x3Dbj = q_wbj.inverse() * (x3Dw - p_wbj);
    Eigen::Vector3d x3Dmj = q_bc.inverse() * (x3Dbj - p_bc);
    Eigen::Vector3d x3Dsj = q_sm * x3Dmj + p_sm;
    double zj = x3Dsj(2);

    residuals = (x3Dsj / zj).head<2>() - pt_sj_td.head<2>();
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce;
        double inv_zj = 1.0f / zj;
        double inv_zj2 = inv_zj * inv_zj;
        reduce << inv_zj, 0, -x3Dsj(0) * inv_zj2,
                  0, inv_zj, -x3Dsj(1) * inv_zj2;

        reduce = sqrt_info * reduce;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = (q_sm * q_bc.inverse() * q_wbj.inverse()).matrix();
            J.rightCols<3>() = - (q_sm * q_bc.inverse() * q_wbj.inverse() * q_wbi).matrix() * Sophus::SO3d::hat(x3Dbi);
            J_pose_i.leftCols<6>() = reduce * J;
            J_pose_i.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[1]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = -(q_sm * q_bc.inverse() * q_wbj.inverse()).matrix();
            J.rightCols<3>() = (q_sm * q_bc.inverse()).matrix() * Sophus::SO3d::hat(x3Dbj);
            J_pose_j.leftCols<6>() = reduce * J;
            J_pose_j.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_bc(jacobians_raw[2]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = (q_sm * q_bc.inverse()).matrix() * ((q_wbj.inverse() * q_wbi).matrix() - Eigen::Matrix3d::Identity());
            J.rightCols<3>() = q_sm.matrix() * (Sophus::SO3d::hat(x3Dmj) - (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc).matrix() * Sophus::SO3d::hat(x3Dmi));

            J_bc.leftCols<6>() = reduce * J;
            J_bc.rightCols<1>().setZero();
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_sm(jacobians_raw[3]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>().setIdentity();
            J.rightCols<3>() = -q_sm.matrix() * Sophus::SO3d::hat(x3Dmj);

            J_sm.leftCols<6>() = reduce * J;
            J_sm.rightCols<1>().setZero();
        }

        if(jacobians_raw[4]) {
            Eigen::Map<Eigen::Vector2d> J_feat(jacobians_raw[4]);
            J_feat = reduce * (q_sm * q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc * (-pt_mi_td / (inv_zi * inv_zi)));
        }

        if(jacobians_raw[5]) {
            Eigen::Map<Eigen::Vector2d> J_td(jacobians_raw[5]);
            J_td = sqrt_info * velocity_j.head<2>() - reduce * (q_sm * q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc * velocity_i / inv_zi);
        }
    }

    return true;
}

SelfProjectionTdFactor::SelfProjectionTdFactor(const Eigen::Vector3d& pt_l_, const Eigen::Vector3d& velocity_l_,
                                               const Eigen::Vector3d& pt_r_, const Eigen::Vector3d& velocity_r_,
                                               double td_0_, double focal_length)
    : td_0(td_0_), pt_l(pt_l_), pt_r(pt_r_), velocity_l(velocity_l_), velocity_r(velocity_r_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
}

bool SelfProjectionTdFactor::Evaluate(double const * const *parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    //parameters [0]: Tsm
    //           [1]: depth
    //           [2]: td
    Eigen::Map<const Sophus::SO3d> q_rl(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_rl(parameters_raw[0] + 4);
    double inv_zl = parameters_raw[1][0];
    double td = parameters_raw[2][0];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Vector3d pt_l_td, pt_r_td;
    // original formular pt_td = pt + td * velocity
    pt_l_td = pt_l - (td - td_0) * velocity_l;
    pt_r_td = pt_r - (td - td_0) * velocity_r;

    Eigen::Vector3d x3Dcl = pt_l_td / inv_zl;
    Eigen::Vector3d x3Dcr = q_rl * x3Dcl + p_rl;
    double zr = x3Dcr(2);

    residuals = (x3Dcr / zr).head<2>() - pt_r_td.head<2>();
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce;
        double inv_zr = 1.0f / zr;
        double inv_zr2 = inv_zr * inv_zr;
        reduce << inv_zr, 0, -x3Dcr(0) * inv_zr2,
                  0, inv_zr, -x3Dcr(1) * inv_zr2;

        reduce = sqrt_info * reduce;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_rl(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>().setIdentity();
            J.rightCols<3>() = -q_rl.matrix() * Sophus::SO3d::hat(x3Dcl);

            J_rl.leftCols<6>() = reduce * J;
            J_rl.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Vector2d> J_feat(jacobians_raw[1]);
            J_feat = reduce * -(q_rl * (pt_l_td / (inv_zl * inv_zl)));
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Vector2d> J_td(jacobians_raw[2]);
            J_td = sqrt_info * velocity_r.head<2>() - reduce * (q_rl * velocity_l / inv_zl);
        }
    }
    return true;
}
