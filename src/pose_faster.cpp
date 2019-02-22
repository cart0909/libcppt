#include "pose_faster.h"

static void PredictNextPose(Eigen::Vector3d& pwb, Sophus::SO3d& qwb, Eigen::Vector3d& vwb,
                     Eigen::Vector3d& gyr_0, Eigen::Vector3d& acc_0, double& t0,
                     const Eigen::Vector3d& gyr_1, const Eigen::Vector3d& acc_1, double t1,
                     const Eigen::Vector3d& ba, const Eigen::Vector3d& bg, const Eigen::Vector3d& gw)
{
    double dt = t1 - t0;
    Eigen::Vector3d un_acc_0 = qwb * (acc_0 - ba) - gw;
    Eigen::Vector3d un_gyr = 0.5 * (gyr_0 + gyr_1) - bg;
    qwb = qwb * Sophus::SO3d::exp(un_gyr * dt);
    Eigen::Vector3d un_acc_1 = qwb * (acc_1 - ba) - gw;
    Eigen::Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
    pwb += dt * vwb + 0.5 * dt * dt * un_acc;
    vwb += dt * un_acc;

    gyr_0 = gyr_1;
    acc_0 = acc_1;
    t0 = t1;
}

PoseFaster::PoseFaster(const Sophus::SO3d& q_bc_, const Eigen::Vector3d& p_bc_, double g_norm)
    : p_bc(p_bc_), q_bc(q_bc_), gw(0, 0, g_norm)
{
    no_pose = true;
    update_pose = true;
}

void PoseFaster::UpdatePoseInfo(const Eigen::Vector3d& p_wb_, const Sophus::SO3d& q_wb_, const Eigen::Vector3d& v_wb_,
                                const Eigen::Vector3d& gyr_0_, const Eigen::Vector3d& acc_0_, double t0_,
                                const Eigen::Vector3d& ba_, const Eigen::Vector3d& bg_)
{
    std::unique_lock<std::mutex> lock(mtx_update);
    if(no_pose)
        no_pose = false;
    update_pose = true;

    p_wb = p_wb_;
    q_wb = q_wb_;
    v_wb = v_wb_;
    gyr_0 = gyr_0_;
    acc_0 = acc_0_;
    pose_t = t0_;
    ba = ba_;
    bg = bg_;

    int index = -1;
    for(int i = 0, n = d_imu_t.size(); i < n; ++i) {
        if(pose_t < d_imu_t[i]) {
            index = i;
            break;
        }
    }

    if(index > 0) {
        d_acc.erase(d_acc.begin(), d_acc.begin() + index);
        d_gyr.erase(d_gyr.begin(), d_gyr.begin() + index);
        d_imu_t.erase(d_imu_t.begin(), d_imu_t.begin() + index);
    }
}

bool PoseFaster::Predict(const Eigen::Vector3d& gyr, const Eigen::Vector3d& acc, double t,
                         Sophus::SE3d& Twc)
{
    std::unique_lock<std::mutex> lock(mtx_update);
    d_acc.emplace_back(acc);
    d_gyr.emplace_back(gyr);
    d_imu_t.emplace_back(t);

    if(no_pose) {
        return false;
    }
    else if(update_pose){
        update_pose = false;
        for(int i = 0, n = d_imu_t.size(); i < n; ++i) {
            PredictNextPose(p_wb, q_wb, v_wb,
                            gyr_0, acc_0, pose_t,
                            d_gyr[i], d_acc[i], d_imu_t[i],
                            ba, bg, gw);
        }
        Twc = Sophus::SE3d(q_wb, p_wb) * Sophus::SE3d(q_bc, p_bc);
    }
    else {
        PredictNextPose(p_wb, q_wb, v_wb,
                        gyr_0, acc_0, pose_t,
                        gyr, acc, t,
                        ba, bg, gw);
        Twc = Sophus::SE3d(q_wb, p_wb) * Sophus::SE3d(q_bc, p_bc);
    }
    return true;
}
