#include "line_projection_factor.h"
#include "util.h"

namespace Plucker {
LineProjectionFactor::LineProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length)
    : spt(spt_), ept(ept_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
}

bool LineProjectionFactor::Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    // parameters [0]: Twb
    //            [1]: Tbc
    //            [2]: Lw
    Eigen::Map<const Sophus::SE3d> Twb(parameters_raw[0]), Tbc(parameters_raw[1]);
    Eigen::Map<const Plucker::Line3d> Lw(parameters_raw[2]);
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Sophus::SE3d Tbw = Twb.inverse(), Tcb = Tbc.inverse();

    Plucker::Line3d Lb = Tbw * Lw;
    Plucker::Line3d Lc = Tcb * Lb;
    Eigen::Vector3d l = Lc.m() / Lc.m().head<2>().norm(); // 2d line in normal plane equal to Lc normal vector
    residuals << l.dot(spt),
                 l.dot(ept);

    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce;
        Eigen::Vector3d mc = Lc.m();
        double m01 = mc.head<2>().norm();
        double inv_m01 = 1.0f / m01, inv_m01_2 = inv_m01 * inv_m01;
        reduce << spt(0)*inv_m01-mc(0)*residuals(0)*inv_m01_2, spt(1)*inv_m01-mc(1)*residuals(0)*inv_m01_2, inv_m01,
                  ept(0)*inv_m01-mc(0)*residuals(1)*inv_m01_2, ept(1)*inv_m01-mc(1)*residuals(1)*inv_m01_2, inv_m01;
        reduce = sqrt_info * reduce;

        Eigen::Matrix3d Rcb = Tbc.so3().inverse().matrix(),
                        Rbw = Twb.so3().inverse().matrix(),
                        Rcw = (Twb.so3() * Tbc.so3()).inverse().matrix();

        Eigen::Vector3d p_bc = Tbc.translation(), p_wb = Twb.translation();
        auto hat = Sophus::SO3d::hat;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jwb(jacobians_raw[0]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = Rcw * hat(Lw.l());
            J.rightCols<3>() = Rcb * (hat(Lb.m()) - hat(p_bc) * hat(Lb.l()));
            Jwb.leftCols<6>() = reduce * J;
            Jwb.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jbc(jacobians_raw[1]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = Rcb * hat(Lb.l());
            J.rightCols<3>() = hat(Lc.m());
            Jbc.leftCols<6>() = reduce * J;
            Jbc.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Jline(jacobians_raw[2]);
            Eigen::Matrix<double, 6, 4> dLw_dTh = Lw.Jacobian();
            Eigen::Matrix<double, 3, 4> dmw_dTh = dLw_dTh.topRows<3>(),
                                        dlw_dTh = dLw_dTh.bottomRows<3>();
            Jline.leftCols<4>() = reduce * Rcb * ((Rbw * dmw_dTh - Rbw * hat(p_wb) * dlw_dTh) - hat(p_bc) * (Rbw * dlw_dTh));
            Jline.rightCols<2>().setZero();
        }
    }
    return true;
}

LineSlaveProjectionFactor::LineSlaveProjectionFactor(const Eigen::Vector3d& spt_, const Eigen::Vector3d& ept_, double focal_length)
    : spt(spt_), ept(ept_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5f;
}

bool LineSlaveProjectionFactor::Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const {
    // parameters [0]: Twb
    //            [1]: Tbc
    //            [2]: Tsm
    //            [3]: Lw
    Eigen::Map<const Sophus::SE3d> Twb(parameters_raw[0]), Tbc(parameters_raw[1]), Tsm(parameters_raw[2]);
    Eigen::Map<const Plucker::Line3d> Lw(parameters_raw[3]);
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Sophus::SE3d Tbw = Twb.inverse(), Tcb = Tbc.inverse();

    Plucker::Line3d Lb = Tbw * Lw;
    Plucker::Line3d Lc = Tcb * Lb;
    Plucker::Line3d Ls = Tsm * Lc;
    Eigen::Vector3d l = Ls.m() / Ls.m().head<2>().norm(); // 2d line in normal plane equal to Lc normal vector

    residuals << l.dot(spt),
                 l.dot(ept);

    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce;
        Eigen::Vector3d ms = Ls.m();
        double m01 = ms.head<2>().norm();
        double inv_m01 = 1.0f / m01, inv_m01_2 = inv_m01 * inv_m01;
        reduce << spt(0)*inv_m01-ms(0)*residuals(0)*inv_m01_2, spt(1)*inv_m01-ms(1)*residuals(0)*inv_m01_2, inv_m01,
                  ept(0)*inv_m01-ms(0)*residuals(1)*inv_m01_2, ept(1)*inv_m01-ms(1)*residuals(1)*inv_m01_2, inv_m01;
        reduce = sqrt_info * reduce;

        Sophus::SO3d q_sm = Tsm.so3(), q_cb = Tbc.so3().inverse(), q_bw = Twb.so3().inverse(),
                     q_sb = q_sm * q_cb,
                     q_sw = q_sb * q_bw;

        Eigen::Matrix3d Rsm = q_sm.matrix(), Rsb = q_sb.matrix(), Rsw = q_sw.matrix(), Rbw = q_bw.matrix();

        const Eigen::Vector3d &p_sm = Tsm.translation(),
                              &p_bc = Tbc.translation(),
                              &p_wb = Twb.translation();

        auto hat = Sophus::SO3d::hat;
        Eigen::Matrix3d lw_hat = hat(Lw.l()),
                        mb_hat = hat(Lb.m()), lb_hat = hat(Lb.l()),
                        mc_hat = hat(Lc.m()), lc_hat = hat(Lc.l()),
                        twb_hat = hat(p_wb), tbc_hat = hat(p_bc), tsm_hat = hat(p_sm);

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jwb(jacobians_raw[0]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = Rsw * lw_hat;
            J.rightCols<3>() = Rsb * (mb_hat - tbc_hat * lb_hat) + tsm_hat * Rsb * lb_hat;
            Jwb.leftCols<6>() = reduce * J;
            Jwb.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jbc(jacobians_raw[1]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = Rsb * lb_hat;
            J.rightCols<3>() = Rsm * mc_hat + tsm_hat * Rsm * lc_hat;
            Jbc.leftCols<6>() = reduce * J;
            Jbc.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jsm(jacobians_raw[2]);
            Eigen::Matrix<double, 3, 6> J;
            J.leftCols<3>() = -hat(q_sm * Lc.l());
            J.rightCols<3>() = -(Rsm * mc_hat + tsm_hat * Rsm * lc_hat);
            Jsm.leftCols<6>() = reduce * J;
            Jsm.rightCols<1>().setZero();
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Jline(jacobians_raw[3]);
            Eigen::Matrix<double, 6, 4> dLw_dTh = Lw.Jacobian();
            Eigen::Matrix<double, 3, 4> dmw_dTh = dLw_dTh.topRows<3>(),
                                        dlw_dTh = dLw_dTh.bottomRows<3>();
            Jline.leftCols<4>() = reduce * (Rsw * dmw_dTh + (tsm_hat * Rsw - Rsb * tbc_hat * Rbw - Rsw * twb_hat) * dlw_dTh);
            Jline.rightCols<2>().setZero();
        }
    }
    return true;
}

}


LineProjectionFactor::LineProjectionFactor(const Eigen::Vector3d& spi_, const Eigen::Vector3d& epi_,
                                           const Eigen::Vector3d& spj_, const Eigen::Vector3d& epj_,
                                           double focal_length)
    : spi(spi_), epi(epi_), spj(spj_), epj(epj_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 3;
}

bool LineProjectionFactor::Evaluate(double const * const* parameters_raw,
                                    double* residuals_raw,
                                    double** jacobians_raw) const
{
    // parameters [0]: frame i
    //            [1]: frame j
    //            [2]: Tbc
    //            [3]: depth P and Q(start point and end point)
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Eigen::Map<const Sophus::SO3d> q_bc(parameters_raw[2]);
    Eigen::Map<const Eigen::Vector3d> p_bc(parameters_raw[2] + 4);
    double P_inv_zi = parameters_raw[3][0];
    double Q_inv_zi = parameters_raw[3][1];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Vector3d Pci = spi / P_inv_zi;
    Eigen::Vector3d Pbi = q_bc * Pci + p_bc;
    Eigen::Vector3d Pw = q_wbi * Pbi + p_wbi;
    Eigen::Vector3d Pbj = q_wbj.inverse() * (Pw - p_wbj);
    Eigen::Vector3d Pcj = q_bc.inverse() * (Pbj - p_bc);
    double P_inv_zj = 1.0f / Pcj(2);

    Eigen::Vector3d Qci = epi / Q_inv_zi;
    Eigen::Vector3d Qbi = q_bc * Qci + p_bc;
    Eigen::Vector3d Qw = q_wbi * Qbi + p_wbi;
    Eigen::Vector3d Qbj = q_wbj.inverse() * (Qw - p_wbj);
    Eigen::Vector3d Qcj = q_bc.inverse() * (Qbj - p_bc);
    double Q_inv_zj = 1.0f / Qcj(2);

    Eigen::Vector3d l = spj.cross(epj);
    l /= l.head<2>().norm();

    residuals << l.dot(Pcj * P_inv_zj),
                 l.dot(Qcj * Q_inv_zj);
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce_Pj, reduce_Qj;
        double P_inv_zj2 = P_inv_zj * P_inv_zj,
               Q_inv_zj2 = Q_inv_zj * Q_inv_zj;
        reduce_Pj << l(0)*P_inv_zj, l(1)*P_inv_zj, -(l(0)*Pcj(0)+l(1)*Pcj(1))*P_inv_zj2,
                                 0,             0,                                    0;
        reduce_Pj = sqrt_info * reduce_Pj;

        reduce_Qj <<             0,             0,                                    0,
                     l(0)*Q_inv_zj, l(1)*Q_inv_zj, -(l(0)*Qcj(0)+l(1)*Qcj(1))*Q_inv_zj2;
        reduce_Qj = sqrt_info * reduce_Qj;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            Eigen::Matrix3_6d Jp, Jq;
            Sophus::SO3d q_cj_w = (q_wbj * q_bc).inverse();
            Sophus::SO3d q_cj_bi = q_cj_w * q_wbi;
            Eigen::Matrix3d R_cj_bi = q_cj_bi.matrix();
            Jp.leftCols<3>() = q_cj_w.matrix();
            Jq.leftCols<3>() = Jp.leftCols<3>();
            Jp.rightCols<3>() = R_cj_bi * -Sophus::SO3d::hat(Pbi);
            Jq.rightCols<3>() = R_cj_bi * -Sophus::SO3d::hat(Qbi);
            J_pose_i.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_pose_i.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[1]);
            Eigen::Matrix3_6d Jp, Jq;
            Sophus::SO3d q_cj_w = (q_wbj * q_bc).inverse();
            Jp.leftCols<3>() = -q_cj_w.matrix();
            Jq.leftCols<3>() = Jp.leftCols<3>();
            Eigen::Matrix3d Rcb = q_bc.inverse().matrix();
            Jp.rightCols<3>() = Rcb * Sophus::SO3d::hat(Pbj);
            Jq.rightCols<3>() = Rcb * Sophus::SO3d::hat(Qbj);
            J_pose_j.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_pose_j.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_bc(jacobians_raw[2]);
            Eigen::Matrix3_6d Jp, Jq;
            Jp.leftCols<3>() = q_bc.inverse().matrix() * ((q_wbj.inverse() * q_wbi).matrix() - Eigen::Matrix3d::Identity());
            Jq.leftCols<3>() = Jp.leftCols<3>();
            Eigen::Matrix3d R_cj_ci = (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc).matrix();
            Jp.rightCols<3>() = Sophus::SO3d::hat(Pcj) - R_cj_ci * Sophus::SO3d::hat(Pci);
            Jq.rightCols<3>() = Sophus::SO3d::hat(Qcj) - R_cj_ci * Sophus::SO3d::hat(Qci);
            J_bc.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_bc.rightCols<1>().setZero();
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J_line(jacobians_raw[3]);
            Sophus::SO3d q_cj_ci = q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc;
            J_line.col(0) = reduce_Pj * (-1.0 / (P_inv_zi * P_inv_zi)) * (q_cj_ci * spi);
            J_line.col(1) = reduce_Qj * (-1.0 / (Q_inv_zi * Q_inv_zi)) * (q_cj_ci * epi);
        }
    }
    return true;
}

LineSlaveProjectionFactor::LineSlaveProjectionFactor(const Eigen::Vector3d& sp_mi_, const Eigen::Vector3d& ep_mi_,
                                                     const Eigen::Vector3d& sp_sj_, const Eigen::Vector3d& ep_sj_,
                                                     double focal_length)
    : sp_mi(sp_mi_), ep_mi(ep_mi_), sp_sj(sp_sj_), ep_sj(ep_sj_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 3;
}

bool LineSlaveProjectionFactor::Evaluate(double const * const* parameters_raw,
                                         double* residuals_raw,
                                         double** jacobians_raw) const
{
    // parameters [0]: frame i
    //            [1]: frame j
    //            [2]: Tbc
    //            [3]: Tsm
    //            [4]: depth P and Q(start point and end point)
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Eigen::Map<const Sophus::SO3d> q_bc(parameters_raw[2]);
    Eigen::Map<const Eigen::Vector3d> p_bc(parameters_raw[2] + 4);
    Eigen::Map<const Sophus::SO3d> q_sm(parameters_raw[3]);
    Eigen::Map<const Eigen::Vector3d> p_sm(parameters_raw[3] + 4);
    double P_inv_zi = parameters_raw[4][0];
    double Q_inv_zi = parameters_raw[4][1];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Vector3d Pmi = sp_mi / P_inv_zi;
    Eigen::Vector3d Pbi = q_bc * Pmi + p_bc;
    Eigen::Vector3d Pw = q_wbi * Pbi + p_wbi;
    Eigen::Vector3d Pbj = q_wbj.inverse() * (Pw - p_wbj);
    Eigen::Vector3d Pmj = q_bc.inverse() * (Pbj - p_bc);
    Eigen::Vector3d Psj = q_sm * Pmj + p_sm;
    double P_inv_zj = 1.0f / Psj(2);

    Eigen::Vector3d Qmi = ep_mi / Q_inv_zi;
    Eigen::Vector3d Qbi = q_bc * Qmi + p_bc;
    Eigen::Vector3d Qw = q_wbi * Qbi + p_wbi;
    Eigen::Vector3d Qbj = q_wbj.inverse() * (Qw - p_wbj);
    Eigen::Vector3d Qmj = q_bc.inverse() * (Qbj - p_bc);
    Eigen::Vector3d Qsj = q_sm * Qmj + p_sm;
    double Q_inv_zj = 1.0f / Qsj(2);

    Eigen::Vector3d l = sp_sj.cross(ep_sj);
    l /= l.head<2>().norm();

    residuals << l.dot(Psj * P_inv_zj),
                 l.dot(Qsj * Q_inv_zj);
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce_Pj, reduce_Qj;
        double P_inv_zj2 = P_inv_zj * P_inv_zj,
               Q_inv_zj2 = Q_inv_zj * Q_inv_zj;
        reduce_Pj << l(0)*P_inv_zj, l(1)*P_inv_zj, -(l(0)*Psj(0)+l(1)*Psj(1))*P_inv_zj2,
                                 0,             0,                                    0;
        reduce_Pj = sqrt_info * reduce_Pj;

        reduce_Qj <<             0,             0,                                    0,
                     l(0)*Q_inv_zj, l(1)*Q_inv_zj, -(l(0)*Qsj(0)+l(1)*Qsj(1))*Q_inv_zj2;
        reduce_Qj = sqrt_info * reduce_Qj;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            Eigen::Matrix3_6d Jp, Jq;
            Sophus::SO3d q_sj_w = q_sm * q_bc.inverse() * q_wbj.inverse();
            Sophus::SO3d q_sj_bi = q_sj_w * q_wbi;
            Eigen::Matrix3d R_sj_bi = q_sj_bi.matrix();
            Jp.leftCols<3>() = q_sj_w.matrix();
            Jq.leftCols<3>() = Jp.leftCols<3>();
            Jp.rightCols<3>() = -R_sj_bi * Sophus::SO3d::hat(Pbi);
            Jq.rightCols<3>() = -R_sj_bi * Sophus::SO3d::hat(Qbi);
            J_pose_i.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_pose_i.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[1]);
            Eigen::Matrix3_6d Jp, Jq;
            Sophus::SO3d q_sb = q_sm * q_bc.inverse();
            Eigen::Matrix3d R_sb = q_sb.matrix();
            Jp.leftCols<3>() = -(q_sb * q_wbj.inverse()).matrix();
            Jq.leftCols<3>() = Jp.leftCols<3>();
            Jp.rightCols<3>() = R_sb * Sophus::SO3d::hat(Pbj);
            Jq.rightCols<3>() = R_sb * Sophus::SO3d::hat(Qbj);
            J_pose_j.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_pose_j.rightCols<1>().setZero();
        }

        if(jacobians_raw[2]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_bc(jacobians_raw[2]);
            Eigen::Matrix3_6d Jp, Jq;
            Jp.leftCols<3>() = (q_sm * q_bc.inverse()).matrix() * ((q_wbj.inverse() * q_wbi).matrix() - Eigen::Matrix3d::Identity());
            Jq.leftCols<3>() = Jp.leftCols<3>();
            Eigen::Matrix3d R_cj_ci = (q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc).matrix();
            Jp.rightCols<3>() = q_sm.matrix() * (Sophus::SO3d::hat(Pmj) - R_cj_ci * Sophus::SO3d::hat(Pmi));
            Jq.rightCols<3>() = q_sm.matrix() * (Sophus::SO3d::hat(Qmj) - R_cj_ci * Sophus::SO3d::hat(Qmi));
            J_bc.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_bc.rightCols<1>().setZero();
        }

        if(jacobians_raw[3]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_sm(jacobians_raw[3]);
            Eigen::Matrix3_6d Jp, Jq;
            Jp.leftCols<3>().setIdentity();
            Jq.leftCols<3>().setIdentity();
            Jp.rightCols<3>() = -q_sm.matrix() * Sophus::SO3d::hat(Pmj);
            Jq.rightCols<3>() = -q_sm.matrix() * Sophus::SO3d::hat(Qmj);
            J_sm.leftCols<6>() = reduce_Pj * Jp + reduce_Qj * Jq;
            J_sm.rightCols<1>().setZero();
        }

        if(jacobians_raw[4]) {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J_line(jacobians_raw[4]);
            Sophus::SO3d q_sj_mi = q_sm * q_bc.inverse() * q_wbj.inverse() * q_wbi * q_bc;
            J_line.col(0) = reduce_Pj * (q_sj_mi * (-sp_mi / (P_inv_zi * P_inv_zi)));
            J_line.col(1) = reduce_Qj * (q_sj_mi * (-ep_mi / (Q_inv_zj * Q_inv_zj)));
        }
    }
    return true;
}

LineSelfProjectionFactor::LineSelfProjectionFactor(const Eigen::Vector3d& sp_l_, const Eigen::Vector3d& ep_l_,
                                                   const Eigen::Vector3d& sp_r_, const Eigen::Vector3d& ep_r_,
                                                   double focal_length)
    : spl(sp_l_), epl(ep_l_), spr(sp_r_), epr(ep_r_)
{
    sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 3;
}

bool LineSelfProjectionFactor::Evaluate(double const * const* parameters_raw,
                                        double* residuals_raw,
                                        double** jacobians_raw) const
{
    Eigen::Map<const Sophus::SO3d> q_rl(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_rl(parameters_raw[0] + 4);
    double P_inv_zl = parameters_raw[1][0];
    double Q_inv_zl = parameters_raw[1][1];
    Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

    Eigen::Vector3d Pcl = spl / P_inv_zl;
    Eigen::Vector3d Pcr = q_rl * Pcl + p_rl;
    double P_inv_zr = 1.0f / Pcr(2);

    Eigen::Vector3d Qcl = epl / Q_inv_zl;
    Eigen::Vector3d Qcr = q_rl * Qcl + p_rl;
    double Q_inv_zr = 1.0f / Qcr(2);

    Eigen::Vector3d l = spr.cross(epr);
    l /= l.head<2>().norm();

    residuals << l.dot(Pcr * P_inv_zr),
                 l.dot(Qcr * Q_inv_zr);
    residuals = sqrt_info * residuals;

    if(jacobians_raw) {
        Eigen::Matrix2_3d reduce_Pr, reduce_Qr;
        double P_inv_zr2 = P_inv_zr * P_inv_zr,
               Q_inv_zr2 = Q_inv_zr * Q_inv_zr;

        reduce_Pr << l(0)*P_inv_zr, l(1)*P_inv_zr, -(l(0)*Pcr(0)+l(1)*Pcr(1))*P_inv_zr2,
                                 0,             0,                                    0;
        reduce_Pr = sqrt_info * reduce_Pr;

        reduce_Qr <<             0,             0,                                    0,
                     l(0)*Q_inv_zr, l(1)*Q_inv_zr, -(l(0)*Qcr(0)+l(1)*Qcr(1))*Q_inv_zr2;
        reduce_Qr = sqrt_info * reduce_Qr;

        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> J_rl(jacobians_raw[0]);
            Eigen::Matrix3_6d Jp, Jq;
            Eigen::Matrix3d Rrl = q_rl.matrix();
            Jp.leftCols<3>().setIdentity();
            Jq.leftCols<3>().setIdentity();
            Jp.rightCols<3>() = -Rrl * Sophus::SO3d::hat(Pcl);
            Jq.rightCols<3>() = -Rrl * Sophus::SO3d::hat(Qcl);
            J_rl.leftCols<6>() = reduce_Pr * Jp + reduce_Qr * Jq;
            J_rl.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 2, 2, Eigen::RowMajor>> J_line(jacobians_raw[1]);
            J_line.col(0) = reduce_Pr * -(q_rl * (spl / (P_inv_zl * P_inv_zl)));
            J_line.col(1) = reduce_Qr * -(q_rl * (epl / (Q_inv_zl * Q_inv_zl)));
        }
    }
    return true;
}
