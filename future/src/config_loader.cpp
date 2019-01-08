#include "config_loader.h"
#include <glog/logging.h>
#include <opencv2/core/eigen.hpp>

ConfigLoader::Param ConfigLoader::Load(const std::string& config_file) {
    Param param;
    cv::FileStorage fs(config_file, cv::FileStorage::READ);

    if(!fs.isOpened()) {
        throw std::runtime_error("file cannot open!");
    }

    param.num_camera = 1;
    param.width.resize(param.num_camera);
    param.height.resize(param.num_camera);
    param.intrinsic_master.resize(param.num_camera);
    param.distortion_master.resize(param.num_camera);
    param.p_bc.resize(param.num_camera);
    param.q_bc.resize(param.num_camera);
    param.b_slave.resize(param.num_camera);
    param.intrinsic_slave.resize(param.num_camera);
    param.distortion_slave.resize(param.num_camera);
    param.p_rl.resize(param.num_camera);
    param.q_rl.resize(param.num_camera);

    param.width[0] = fs["image_width"];
    param.height[0] = fs["image_height"];
    fs["intrinsics0"] >> param.intrinsic_master[0];
    fs["distortion_coefficients0"] >> param.distortion_master[0];
    param.b_slave[0] = true;
    fs["intrinsics1"] >> param.intrinsic_slave[0];
    fs["distortion_coefficients1"] >> param.distortion_slave[0];

    cv::Mat cv_T;
    Eigen::Matrix4d eigen_T;
    Eigen::Vector3d tmp_p;
    Eigen::Quaterniond tmp_q;
    fs["T_BC0"] >> cv_T;
    cv::cv2eigen(cv_T, eigen_T);
    tmp_p = eigen_T.block<3, 1>(0, 3);
    tmp_q = eigen_T.block<3, 3>(0, 0);
    tmp_q.normalize();
    param.p_bc[0] = tmp_p;
    param.q_bc[0].setQuaternion(tmp_q);

    fs["T_BC1"] >> cv_T;
    cv::cv2eigen(cv_T, eigen_T);
    tmp_p = eigen_T.block<3, 1>(0, 3); // p_bc1
    tmp_q = eigen_T.block<3, 3>(0, 0); // q_bc1
    tmp_q.normalize();
    Sophus::SE3d Tbl(param.q_bc[0], param.p_bc[0]);
    Sophus::SE3d Tbr(tmp_q, tmp_p);
    Sophus::SE3d Trl = Tbr.inverse() * Tbl;
    param.p_rl[0] = Trl.translation();
    param.q_rl[0] = Trl.so3();

    param.acc_noise = fs["acc_n"];
    param.gyr_noise = fs["gyr_n"];
    param.acc_bias_noise = fs["acc_w"];
    param.gyr_bias_noise = fs["gyr_w"];
    param.gravity_magnitude = fs["g_norm"];

    Log(param);
    fs.release();
    return param;
}

void ConfigLoader::Log(const Param& param) {
    LOG(INFO) << "number of camera: " << param.num_camera;
    for(int i = 0; i < param.num_camera; ++i) {
        LOG(INFO) << "cam" << i <<" image size: " << cv::Size(param.width[i], param.height[i]);
        LOG(INFO) << "cam" << i << " intrinsic: "
                  << param.intrinsic_master[i][0] << " "
                  << param.intrinsic_master[i][1] << " "
                  << param.intrinsic_master[i][2] << " "
                  << param.intrinsic_master[i][3];

        LOG(INFO) << "cam" << i << " distortion: "
                  << param.distortion_master[i][0] << " "
                  << param.distortion_master[i][1] << " "
                  << param.distortion_master[i][2] << " "
                  << param.distortion_master[i][3];

        LOG(INFO) << "cam" << i << " p_bc: " << param.p_bc[i](0) << " " <<
                                                param.p_bc[i](1) << " " <<
                                                param.p_bc[i](2);
        LOG(INFO) << "cam" << i << " q_bc: " <<
                     param.q_bc[i].unit_quaternion().w() << " " <<
                     param.q_bc[i].unit_quaternion().x() << " " <<
                     param.q_bc[i].unit_quaternion().y() << " " <<
                     param.q_bc[i].unit_quaternion().z();

        if(!param.b_slave[i])
            continue;

        LOG(INFO) << "cam" << i << "r intrinsic: "
                  << param.intrinsic_slave[i][0] << " "
                  << param.intrinsic_slave[i][1] << " "
                  << param.intrinsic_slave[i][2] << " "
                  << param.intrinsic_slave[i][3];

        LOG(INFO) << "cam" << i << "r distortion: "
                  << param.distortion_slave[i][0] << " "
                  << param.distortion_slave[i][1] << " "
                  << param.distortion_slave[i][2] << " "
                  << param.distortion_slave[i][3];

        LOG(INFO) << "cam" << i << "r p_rl: " << param.p_rl[i](0) << " " <<
                                                 param.p_rl[i](1) << " " <<
                                                 param.p_rl[i](2);
        LOG(INFO) << "cam" << i << "r q_rl: " <<
                     param.q_rl[i].unit_quaternion().w() << " " <<
                     param.q_rl[i].unit_quaternion().x() << " " <<
                     param.q_rl[i].unit_quaternion().y() << " " <<
                     param.q_rl[i].unit_quaternion().z();
    }

    LOG(INFO) << "acc_noise: " << param.acc_noise;
    LOG(INFO) << "gyr_noise: " << param.gyr_noise;
    LOG(INFO) << "acc_bias_noise: " << param.acc_bias_noise;
    LOG(INFO) << "gyr_bias_noise: " << param.gyr_bias_noise;
    LOG(INFO) << "gravity_magnitude: " << param.gravity_magnitude;
}
