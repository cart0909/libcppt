#include "lidarFactor.h"

LidarLineFactorNeedDistort::LidarLineFactorNeedDistort(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& last_point_a_,
                                                       const Eigen::Vector3d& last_point_b_, const Sophus::SE3d& body_T_lidar_)
    :curr_point(curr_point_), last_point_a(last_point_a_), last_point_b(last_point_b_), body_T_lidar(body_T_lidar_){

}


bool LidarLineFactorNeedDistort::Evaluate(double const * const* parameters_raw,
                                          double* residuals_raw,
                                          double** jacobians_raw) const{
    Eigen::Map<const Sophus::SO3d> wmap_q_bodyj(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> wmap_p_bodyj(parameters_raw[0] + 4);
    Eigen::Vector3d lp;
    Eigen::Map<Eigen::Vector3d> residuals(residuals_raw);
    Sophus::SO3d wmap_q_lidarj = wmap_q_bodyj * body_T_lidar.so3();
    Eigen::Vector3d wmap_p_lidarj = wmap_q_bodyj * body_T_lidar.translation() + wmap_p_bodyj;
    lp = wmap_q_lidarj * curr_point + wmap_p_lidarj;

    //estimate residual
    Eigen::Vector3d nu = (lp - last_point_a).cross(lp - last_point_b);
    Eigen::Vector3d de = last_point_a - last_point_b;
    Eigen::Vector3d lp_lpa = curr_point - last_point_a;
    Eigen::Vector3d lp_lpb = curr_point - last_point_b;
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
            Eigen::Map<Eigen::Matrix<double, 3, 7, Eigen::RowMajor>> J_lp_pose(jacobians_raw[0]);
            Eigen::Matrix3_6d J;
            Eigen::Matrix<double, 3, 3> I;
            I.setIdentity();
            J.leftCols<3>() = I;
            J.rightCols<3>() = -wmap_q_bodyj.matrix() * ( Sophus::SO3d::hat(body_T_lidar.so3() * curr_point)  +  Sophus::SO3d::hat(wmap_p_bodyj));
            J_lp_pose.leftCols<6>() = fijk_lp * J;
            J_lp_pose.rightCols<1>().setZero();
        }
    }

    return true;
}

LidarPlaneFactorNeedDistort::LidarPlaneFactorNeedDistort(const Eigen::Vector3d& curr_point_, const Eigen::Vector3d& plane_unit_norm_,
                                                         const Sophus::SE3d& body_T_lidar_, double negative_OA_dot_norm_):
    curr_point(curr_point_), plane_unit_norm(plane_unit_norm_), body_T_lidar(body_T_lidar_), negative_OA_dot_norm(negative_OA_dot_norm_)
{


}

bool LidarPlaneFactorNeedDistort::Evaluate(double const * const* parameters_raw, double* residuals_raw, double** jacobians_raw) const{
    Eigen::Map<const Sophus::SO3d> wmap_q_bodyj(parameters_raw[0]);
    Eigen::Map<const Eigen::Vector3d> wmap_p_bodyj(parameters_raw[0] + 4);
    Eigen::Vector3d lp;
    Sophus::SO3d wmap_q_lidarj = wmap_q_bodyj * body_T_lidar.so3();
    Eigen::Vector3d wmap_p_lidarj = wmap_q_bodyj * body_T_lidar.translation() + wmap_p_bodyj;

    lp = wmap_q_lidarj * curr_point + wmap_p_lidarj;
    residuals_raw[0] = plane_unit_norm.dot(lp) + (negative_OA_dot_norm);
    if(jacobians_raw){
        Eigen::Matrix<double, 1, 3> plane_lp;
        plane_lp << plane_unit_norm.x(), plane_unit_norm.y(), plane_unit_norm.z();
      if(jacobians_raw[0]) {
          Eigen::Map<Eigen::Matrix<double, 1, 7, Eigen::RowMajor>> J_lp_pose(jacobians_raw[0]);
          Eigen::Matrix3_6d J;
          Eigen::Matrix<double, 3, 3> I;
          I.setIdentity();
          J.leftCols<3>() = I;
          J.rightCols<3>() = -wmap_q_bodyj.matrix() * ( Sophus::SO3d::hat(body_T_lidar.so3() * curr_point)  +  Sophus::SO3d::hat(wmap_p_bodyj));
          J_lp_pose.leftCols<6>() = plane_lp * J;
          J_lp_pose.rightCols<1>().setZero();
      }
    }

}
