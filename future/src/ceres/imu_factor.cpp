#include "imu_factor.h"

enum {
    I_RIJ = 0,
    I_VIJ = 3,
    I_PIJ = 6,
    I_BA = 9,
    I_BG = 12,

    I_RI = 0,
    I_PI = 3,

    I_RJ = 0,
    I_PJ = 3,

    I_VI = 0,
    I_BAI = 3,
    I_BGI = 6,

    I_VJ = 0,
    I_BAJ = 3,
    I_BGJ = 6
};

ImuFactor::ImuFactor(ImuPreintegrationPtr preintegration_, const Eigen::Vector3d& gw_,
                     const Eigen::Matrix6d& inv_cov_acc_gyr_bias)
    : preintegration(preintegration_), gw(gw_)
{
    sqrt_info.setIdentity();
    // FIXME
    sqrt_info.block<9, 9>(0, 0) = Eigen::LLT<Eigen::Matrix9d>(preintegration->mCovariance.inverse()).matrixL().transpose();
    sqrt_info.block<6, 6>(9, 9) = Eigen::LLT<Eigen::Matrix6d>(inv_cov_acc_gyr_bias).matrixL().transpose();
}

bool ImuFactor::Evaluate(double const *const *parameters_raw,
                         double *residuals_raw, double **jacobians_raw) const {
    Eigen::Map<const Sophus::SO3d> qwbi(parameters_raw[0]);
    Sophus::SO3d qbi_w = qwbi.inverse();
    Eigen::Map<const Eigen::Vector3d> pwbi(parameters_raw[0] + 4);
    Eigen::Map<const Eigen::Vector3d> vwbi(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> bai(parameters_raw[1] + 3);
    Eigen::Map<const Eigen::Vector3d> bgi(parameters_raw[1] + 6);

    Eigen::Map<const Sophus::SO3d> qwbj(parameters_raw[2]);
    Sophus::SO3d qbj_w = qwbj.inverse();
    Eigen::Map<const Eigen::Vector3d> pwbj(parameters_raw[2] + 4);
    Eigen::Map<const Eigen::Vector3d> vwbj(parameters_raw[3]);
    Eigen::Map<const Eigen::Vector3d> baj(parameters_raw[3] + 3);
    Eigen::Map<const Eigen::Vector3d> bgj(parameters_raw[3] + 6);

    Eigen::Map<Eigen::Vector15d> residuals(residuals_raw);
    Eigen::Map<Eigen::Vector3d> r_delRij(residuals_raw);
    Eigen::Map<Eigen::Vector3d> r_delVij(residuals_raw + 3);
    Eigen::Map<Eigen::Vector3d> r_delPij(residuals_raw + 6);
    Eigen::Map<Eigen::Vector3d> r_ba(residuals_raw + 9);
    Eigen::Map<Eigen::Vector3d> r_bg(residuals_raw + 12);

    Eigen::Vector3d delta_bg = bgi - preintegration->mbg;
    Eigen::Vector3d delta_ba = bai - preintegration->mba;

    // bias update cause the preintegration need to re integrate the state
    // it cost much time to do
    // the simple way is using taylor approximation
    Sophus::SO3d delRij = preintegration->mdelRij * Sophus::SO3d::exp(preintegration->mJR_bg * delta_bg);
    Eigen::Vector3d delVij = preintegration->mdelVij + preintegration->mJV_bg * delta_bg + preintegration->mJV_ba * delta_ba;
    Eigen::Vector3d delPij = preintegration->mdelPij + preintegration->mJP_bg * delta_bg + preintegration->mJP_ba * delta_ba;

    double dtij = preintegration->mdel_tij, dtij2 = dtij * dtij;

    r_delRij = (delRij.inverse() * qbi_w * qwbj).log();
    r_delVij = qbi_w * (vwbj - vwbi - gw * dtij) - delVij;
    r_delPij = qbi_w * (pwbj - pwbi - vwbi * dtij - 0.5 * gw * dtij2) - delPij;
    r_ba = baj - bai;
    r_bg = bgj - bgi;

    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix3d Jr_r_delRij_inv = Sophus::JacobianRInv(r_delRij);
        Eigen::Matrix3d I3x3 = Eigen::Matrix3d::Identity();
        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            J_pose_i.setZero();
            J_pose_i.block<3, 3>(I_RIJ, I_RI) = -Jr_r_delRij_inv * (qbj_w * qwbi).matrix();
            J_pose_i.block<3, 3>(I_VIJ, I_RI) = Sophus::SO3d::hat(qbi_w * (vwbj - vwbi - gw * dtij));
            J_pose_i.block<3, 3>(I_PIJ, I_RI) = Sophus::SO3d::hat(qbi_w * (pwbj - pwbi - vwbi * dtij - 0.5 * gw * dtij2));
            J_pose_i.block<3, 3>(I_PIJ, I_PI) = -Eigen::Matrix3d::Identity();
            J_pose_i = sqrt_info * J_pose_i;
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_vb_i(jacobians_raw[1]);
            J_vb_i.setZero();
            J_vb_i.block<3, 3>(I_RIJ, I_BGI) = -Jr_r_delRij_inv * Sophus::SO3d::exp(r_delRij).inverse().matrix() *
                    Sophus::JacobianR(preintegration->mJR_bg * delta_bg) * preintegration->mJR_bg;
            J_vb_i.block<3, 3>(I_VIJ, I_VI) = -qbi_w.matrix();
            J_vb_i.block<3, 3>(I_VIJ, I_BAI) = -preintegration->mJV_ba;
            J_vb_i.block<3, 3>(I_VIJ, I_BGI) = -preintegration->mJV_bg;
            J_vb_i.block<3, 3>(I_PIJ, I_VI) = -qbi_w.matrix() * dtij;
            J_vb_i.block<3, 3>(I_PIJ, I_BAI) = -preintegration->mJP_ba;
            J_vb_i.block<3, 3>(I_PIJ, I_BGI) = -preintegration->mJP_bg;
            J_vb_i.block<3, 3>(I_BA, I_BAI) = -I3x3;
            J_vb_i.block<3, 3>(I_BG, I_BGI) = -I3x3;
            J_vb_i = sqrt_info * J_vb_i;
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[2]);
            J_pose_j.setZero();
            J_pose_j.block<3, 3>(I_RIJ, I_RJ) = Jr_r_delRij_inv;
            J_pose_j.block<3, 3>(I_PIJ, I_PJ) = (qbi_w * qwbj).matrix();
            J_pose_j = sqrt_info * J_pose_j;
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> J_vb_j(jacobians_raw[3]);
            J_vb_j.setZero();
            J_vb_j.block<3, 3>(I_VIJ, I_VJ) = qbi_w.matrix();
            J_vb_j.block<3, 3>(I_BA, I_BAJ) = I3x3;
            J_vb_j.block<3, 3>(I_BG, I_BGJ) = I3x3;
            J_vb_j = sqrt_info * J_vb_j;
        }
    }

    return true;
}
