#include "lidar_factor.h"

LidarEdgeFactor::LidarEdgeFactor(const Eigen::Vector3d& curr_point_,
                                 const Eigen::Vector3d& last_point_a_,
                                 const Eigen::Vector3d& last_point_b_,
                                 const Sophus::SE3d& Tlb_, double s_, double noise_):
    curr_point(curr_point_), last_point_a(last_point_a_),
    last_point_b(last_point_b_), Tlb(Tlb_), s(1){
    //TODO::add covariance
    sqrt_info = (40) * Eigen::Matrix3d::Identity();
}

bool LidarEdgeFactor::Evaluate(double const * const* parameters_raw,
                               double* residuals_raw,
                               double** jacobians_raw) const{
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Eigen::Map<Eigen::Vector3d> residuals(residuals_raw);
    Sophus::SE3d Twbi(q_wbi, p_wbi);
    Sophus::SE3d Twbj(q_wbj, p_wbj);
    Eigen::Vector3d lp;
    Sophus::SE3d Tbl = Tlb.inverse();
    Eigen::Vector3d Xbj = Tbl * curr_point;
    Eigen::Vector3d Xbi = Twbi.inverse() * Twbj * Xbj;
    lp =  Tlb * Xbi;
    //estimate residual
    Eigen::Vector3d lp_lpa = lp - last_point_a;
    Eigen::Vector3d lp_lpb = lp - last_point_b;
    Eigen::Vector3d nu = (lp_lpa).cross(lp_lpb);
    Eigen::Vector3d de = last_point_a - last_point_b;
    residuals.x() = nu.x() / de.norm();
    residuals.y() = nu.y() / de.norm();
    residuals.z() = nu.z() / de.norm();
    residuals = sqrt_info * residuals;

    if(jacobians_raw){
        Eigen::Matrix<double, 3, 3> fijk_lp;
        fijk_lp<<                0          ,    (lp_lpb.z()) - (lp_lpa.z()),    (lp_lpa.y()) - (lp_lpb.y()),
                (lp_lpa.z()) - (lp_lpb.z()),                 0             ,    (lp_lpb.x()) - (lp_lpa.x()),
                (lp_lpb.y()) - (lp_lpa.y()),    (lp_lpa.x()) - (lp_lpb.x()),                 0;
        fijk_lp = fijk_lp / de.norm();
        fijk_lp = sqrt_info * fijk_lp;
        //TODO::point to line jacobian
        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = -(Tlb.so3() * q_wbi.inverse()).matrix();
            J.rightCols<3>() =  Tlb.so3().matrix() * Sophus::SO3d::hat(Xbi);
            J_pose_i.leftCols<6>() = fijk_lp * J;
            J_pose_i.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[1]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = (Tlb.so3() * q_wbi.inverse()).matrix();
            J.rightCols<3>() = -(Tlb.so3() * q_wbi.inverse() * q_wbj).matrix() * Sophus::SO3d::hat(Xbj);
            J_pose_j.leftCols<6>() = fijk_lp * J;
            J_pose_j.rightCols<1>().setZero();
        }
    }
    return true;
}

LidarPlaneFactor::LidarPlaneFactor(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                                   const Eigen::Vector3d& last_point_b_, const Eigen::Vector3d& last_point_d_,
                                   const Sophus::SE3d& Tlb_, double s_, double noise_):
    curr_point(curr_point_), last_point_a(last_point_a_),
    last_point_b(last_point_b_), last_point_d(last_point_d_), Tlb(Tlb_), s(1)
{
    ljm_norm = (last_point_a - last_point_b).cross(last_point_a - last_point_d);
    ljm_norm.normalize();
    noise = 30;
}

bool LidarPlaneFactor::Evaluate(double const * const* parameters_raw,
                                double* residuals_raw,
                                double** jacobians_raw) const{
    Eigen::Map<const Sophus::SO3d> q_wbi(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_wbi(parameters_raw[0] + 4);
    Eigen::Map<const Sophus::SO3d> q_wbj(parameters_raw[1]);
    Eigen::Map<const Eigen::Vector3d> p_wbj(parameters_raw[1] + 4);
    Sophus::SE3d Twbi(q_wbi, p_wbi);
    Sophus::SE3d Twbj(q_wbj, p_wbj);
    Eigen::Vector3d lp;
    Sophus::SE3d Tbl = Tlb.inverse();
    Eigen::Vector3d Xbj = Tbl * curr_point;
    Eigen::Vector3d Xw = Twbj * Xbj;
    Eigen::Vector3d Xbi = Twbi.inverse() * Xw;
    lp =  Tlb * Xbi;
    //estimate residual
    residuals_raw[0] = (lp - last_point_a).dot(ljm_norm);
    residuals_raw[0] = noise * residuals_raw[0];
    if(jacobians_raw){
        Eigen::Matrix<double, 1, 3> plane_lp;
        plane_lp << ljm_norm.x(), ljm_norm.y(), ljm_norm.z();
        plane_lp = noise * plane_lp;
        //TODO::point to line jacobian
        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_pose_i(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = -(Tlb.so3() * q_wbi.inverse()).matrix();
            J.rightCols<3>() =  Tlb.so3().matrix() * Sophus::SO3d::hat(Xbi);
            J_pose_i.leftCols<6>() = plane_lp * J;
            J_pose_i.rightCols<1>().setZero();
        }

        if(jacobians_raw[1]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_pose_j(jacobians_raw[1]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = (Tlb.so3() * q_wbi.inverse()).matrix();
            J.rightCols<3>() = -(Tlb.so3() * q_wbi.inverse() * q_wbj).matrix() * Sophus::SO3d::hat(Xbj);
            J_pose_j.leftCols<6>() = plane_lp * J;
            J_pose_j.rightCols<1>().setZero();
        }
    }
    return true;
}


LidarEdgeFactorIJ::LidarEdgeFactorIJ(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                                     const Eigen::Vector3d& last_point_b_, double s_):
    curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), s(1){

}

bool LidarEdgeFactorIJ::Evaluate(double const * const* parameters_raw,
                                 double* residuals_raw,
                                 double** jacobians_raw) const{
    Eigen::Map<const Sophus::SO3d> q_ij(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_ij(parameters_raw[0] + 4);
    Eigen::Vector3d lp;
    lp = q_ij * curr_point + p_ij;
    Eigen::Vector3d lp_lpa = lp - last_point_a;
    Eigen::Vector3d lp_lpb = lp - last_point_b;
    Eigen::Matrix<double, 3, 1> nu = (lp_lpa).cross(lp_lpb);
    Eigen::Matrix<double, 3, 1> de = last_point_a - last_point_b;
    Eigen::Map<Eigen::Vector3d> residuals(residuals_raw);
    residuals.x() = nu.x() / de.norm();
    residuals.y() = nu.y() / de.norm();
    residuals.z() = nu.z() / de.norm();
    if(jacobians_raw){
        Eigen::Matrix<double, 3, 3> fijk_lp;
        fijk_lp<<                0          ,    (lp_lpb.z()) - (lp_lpa.z()),    (lp_lpa.y()) - (lp_lpb.y()),
                (lp_lpa.z()) - (lp_lpb.z()),                 0             ,    (lp_lpb.x()) - (lp_lpa.x()),
                (lp_lpb.y()) - (lp_lpa.y()),    (lp_lpa.x()) - (lp_lpb.x()),                 0;
        fijk_lp = fijk_lp / de.norm();
        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_pose(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = Eigen::Matrix3d::Identity();
            J.rightCols<3>() = (-q_ij.matrix() * Sophus::SO3d::hat(curr_point));
            J_pose.leftCols<6>() = fijk_lp * J;
            J_pose.rightCols<1>().setZero();
        }
    }

    return true;
}

LidarPlaneFactorIJ::LidarPlaneFactorIJ(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                                       const Eigen::Vector3d& last_point_b_, const Eigen::Vector3d& last_point_d_, double s_):
    curr_point(curr_point_), last_point_a(last_point_a_),
    last_point_b(last_point_b_), last_point_d(last_point_d_), s(1){
    ljm_norm = (last_point_a - last_point_b).cross(last_point_a - last_point_d);
    ljm_norm.normalize();
}

bool LidarPlaneFactorIJ::Evaluate(double const * const* parameters_raw,
                                  double* residuals_raw,
                                  double** jacobians_raw) const{
    Eigen::Map<const Sophus::SO3d> q_ij(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> p_ij(parameters_raw[0] + 4);
    Eigen::Vector3d lp;
    lp = q_ij * curr_point + p_ij;

    residuals_raw[0] = (lp - last_point_a).dot(ljm_norm);
    if(jacobians_raw){
        Eigen::Matrix<double, 1, 3> plane_lp;
        plane_lp << ljm_norm.x(), ljm_norm.y(), ljm_norm.z();
        if(jacobians_raw[0]) {
            Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_pose(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            J.leftCols<3>() = Eigen::Matrix3d::Identity();
            J.rightCols<3>() =   (-q_ij.matrix() * Sophus::SO3d::hat(curr_point));
            J_pose.leftCols<6>() = plane_lp * J;
            J_pose.rightCols<1>().setZero();
        }
    }

    return true;
}
