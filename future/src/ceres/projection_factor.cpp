#include "projection_factor.h"

ProjectionFactor::ProjectionFactor(const Eigen::Vector3d& pt_i_, const Eigen::Vector3d& pt_j_,
                                   const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_,
                                   double focal_length)
    : pt_i(pt_i_), pt_j(pt_j_), q_bc(q_bc_), p_bc(p_bc_)
{
    //1.5 pixel
    sqrt_info = focal_length / 1.5 * Eigen::Matrix2d::Identity();
}

bool ProjectionFactor::Evaluate(double const * const* parameters_raw,
                                double* residuals_raw,
                                double** jacobians_raw) const {
    //parameters [0]: frame i
    //           [1]: frame j
    //           [2]: depth
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);
    double inv_zi = parameters_raw[2][0];

    Eigen::Vector3d x3Dci = pt_i / inv_zi;
    Eigen::Vector3d x3Dbi = q_bc * x3Dci + p_bc;
    Eigen::Vector3d x3Dw = q_wbi * x3Dbi + p_wbi;
    Eigen::Vector3d x3Dbj = q_wbj.inverse() * (x3Dw - p_wbj);
    Eigen::Vector3d x3Dcj = q_bc.inverse() * (x3Dbj - p_bc);
    double zj = x3Dcj(2);
    residuals = (x3Dcj / zj).head<2>() - pt_j.head<2>();
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
            // TODO
            J_pose_j.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Vector2d> J_feat(jacobians_raw[2]);
            J_feat = reduce * (-1.0 / (inv_zi * inv_zi)) *
                    (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc * pt_i);
        }
    }

    return true;
}


SelfProjectionFactor::SelfProjectionFactor(const Eigen::Vector3d& pt_l_, const Eigen::Vector3d& pt_r_,
                                           const Sophus::SO3d& q_rl_, const Eigen::Vector3d& p_rl_,
                                           double focal_length)
    : pt_l(pt_l_), pt_r(pt_r_), q_rl(q_rl_), p_rl(p_rl_)
{
    //1.5 pixel
    sqrt_info = focal_length / 1.5 * Eigen::Matrix2d::Identity();
}

bool SelfProjectionFactor::Evaluate(double const * const* parameters_raw,
                                    double* residuals_raw,
                                    double** jacobians_raw) const {
    double inv_zl = parameters_raw[0][0];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);
    Eigen::Vector3d x3Dcl = pt_l / inv_zl;
    Eigen::Vector3d x3Dcr = q_rl * x3Dcl + p_rl;
    double zr = x3Dcr(2);

    residuals = (x3Dcr / zr).head<2>() - pt_r.head<2>();
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {

    }

    return true;
}
