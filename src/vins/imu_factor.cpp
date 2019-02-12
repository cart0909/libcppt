#include "vins/imu_factor.h"

IMUFactor::IMUFactor(IntegrationBasePtr _pre_integration, const Eigen::Vector3d& Gw_) : pre_integration(_pre_integration), Gw(Gw_)
{
}

bool IMUFactor::Evaluate(double const *const *parameters, double *residuals, double **jacobians) const
{
    Eigen::Map<const Sophus::SO3d> Qwi(parameters[0]), Qwj(parameters[2]);
    Eigen::Map<const Eigen::Vector3d> Pwi(parameters[0] + 4), Pwj(parameters[2] + 4),
                                      Vi(parameters[1]), Bai(parameters[1] + 3), Bgi(parameters[1] + 6),
                                      Vj(parameters[3]), Baj(parameters[3] + 3), Bgj(parameters[3] + 6);

    Eigen::Map<Eigen::Vector15d> residual(residuals);
    residual = pre_integration->evaluate(Pwi, Qwi, Vi, Bai, Bgi,
                                         Pwj, Qwj, Vj, Baj, Bgj, Gw);

    Eigen::Matrix15d sqrt_info = Eigen::LLT<Eigen::Matrix<double, 15, 15>>(pre_integration->covariance.inverse()).matrixL().transpose();
    //sqrt_info.setIdentity();
    residual = sqrt_info * residual;

    if (jacobians)
    {
        double sum_dt = pre_integration->sum_dt;
        Eigen::Matrix3d dp_dba = pre_integration->jacobian.template block<3, 3>(O_P, O_BA);
        Eigen::Matrix3d dp_dbg = pre_integration->jacobian.template block<3, 3>(O_P, O_BG);

        Eigen::Matrix3d dq_dbg = pre_integration->jacobian.template block<3, 3>(O_R, O_BG);

        Eigen::Matrix3d dv_dba = pre_integration->jacobian.template block<3, 3>(O_V, O_BA);
        Eigen::Matrix3d dv_dbg = pre_integration->jacobian.template block<3, 3>(O_V, O_BG);

        Sophus::SO3d Qiw = Qwi.inverse(), Qjw = Qwj.inverse();
        Eigen::Matrix3d Riw = Qiw.matrix(), I3x3 = Eigen::Matrix3d::Identity();
        Sophus::SO3d corrected_delta_q = pre_integration->delta_q * Sophus::SO3d::exp(dq_dbg * (Bgi - pre_integration->linearized_bg));

        if (pre_integration->jacobian.maxCoeff() > 1e8 || pre_integration->jacobian.minCoeff() < -1e8)
        {
            ROS_WARN("numerical unstable in preintegration");
            //std::cout << pre_integration->jacobian << std::endl;
        }

        if (jacobians[0])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_i(jacobians[0]);
            jacobian_pose_i.setZero();

            jacobian_pose_i.block<3, 3>(O_P, O_P) = -Riw;
            jacobian_pose_i.block<3, 3>(O_P, O_R) = Sophus::SO3d::hat(Qiw * (0.5 * Gw * sum_dt * sum_dt + Pwj - Pwi - Vi * sum_dt));

            jacobian_pose_i.block<3, 3>(O_R, O_R) = -(Sophus::Qleft(Qjw * Qwi) * Sophus::Qright(corrected_delta_q)).bottomRightCorner<3, 3>();

            jacobian_pose_i.block<3, 3>(O_V, O_R) = Sophus::SO3d::hat(Qiw * (Gw * sum_dt + Vj - Vi));

            jacobian_pose_i = sqrt_info * jacobian_pose_i;

            if (jacobian_pose_i.maxCoeff() > 1e8 || jacobian_pose_i.minCoeff() < -1e8)
            {
                ROS_WARN("numerical unstable in preintegration");
                //std::cout << sqrt_info << std::endl;
                //ROS_BREAK();
            }
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_i(jacobians[1]);
            jacobian_speedbias_i.setZero();
            jacobian_speedbias_i.block<3, 3>(O_P, O_V - O_V) = -Riw * sum_dt;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BA - O_V) = -dp_dba;
            jacobian_speedbias_i.block<3, 3>(O_P, O_BG - O_V) = -dp_dbg;

            jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Sophus::Qleft(Qjw * Qwi * corrected_delta_q).bottomRightCorner<3, 3>() * dq_dbg;
            //jacobian_speedbias_i.block<3, 3>(O_R, O_BG - O_V) = -Sophus::Qleft(Qjw * Qwi * pre_integration->delta_q).bottomRightCorner<3, 3>() * dq_dbg;

            jacobian_speedbias_i.block<3, 3>(O_V, O_V - O_V) = -Riw;
            jacobian_speedbias_i.block<3, 3>(O_V, O_BA - O_V) = -dv_dba;
            jacobian_speedbias_i.block<3, 3>(O_V, O_BG - O_V) = -dv_dbg;

            jacobian_speedbias_i.block<3, 3>(O_BA, O_BA - O_V) = -I3x3;

            jacobian_speedbias_i.block<3, 3>(O_BG, O_BG - O_V) = -I3x3;

            jacobian_speedbias_i = sqrt_info * jacobian_speedbias_i;
        }
        if (jacobians[2])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 7, Eigen::RowMajor>> jacobian_pose_j(jacobians[2]);
            jacobian_pose_j.setZero();

            jacobian_pose_j.block<3, 3>(O_P, O_P) = Riw;

            jacobian_pose_j.block<3, 3>(O_R, O_R) = Sophus::Qleft(corrected_delta_q.inverse() * Qiw * Qwj).bottomRightCorner<3, 3>();

            jacobian_pose_j = sqrt_info * jacobian_pose_j;
        }
        if (jacobians[3])
        {
            Eigen::Map<Eigen::Matrix<double, 15, 9, Eigen::RowMajor>> jacobian_speedbias_j(jacobians[3]);
            jacobian_speedbias_j.setZero();

            jacobian_speedbias_j.block<3, 3>(O_V, O_V - O_V) = Riw;

            jacobian_speedbias_j.block<3, 3>(O_BA, O_BA - O_V) = I3x3;

            jacobian_speedbias_j.block<3, 3>(O_BG, O_BG - O_V) = I3x3;

            jacobian_speedbias_j = sqrt_info * jacobian_speedbias_j;
        }
    }

    return true;
}
