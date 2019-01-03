#include "vio_system.h"
#include <ros/ros.h>
#include <opencv2/core/eigen.hpp>

VIOSystem::VIOSystem(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);

    int width;
    int height;
    height = fs["image_height"];
    width = fs["image_width"];

    cv::Size image_size(width, height);
    cv::Mat Tbc0, Tbc1, Tbi;
    fs["T_BC0"] >> Tbc0;
    fs["T_BC1"] >> Tbc1;
    fs["T_BI"] >> Tbi;

    std::vector<double> intrinsics0, intrinsics1;
    std::vector<double> distortion_coefficients0, distortion_coefficients1;
    fs["intrinsics0"] >> intrinsics0;
    fs["distortion_coefficients0"] >> distortion_coefficients0;
    fs["intrinsics1"] >> intrinsics1;
    fs["distortion_coefficients1"] >> distortion_coefficients1;

    cv::Mat K0, K1, D0, D1;
    K0 = (cv::Mat_<double>(3, 3) << intrinsics0[0], 0, intrinsics0[2],
                                    0, intrinsics0[1], intrinsics0[3],
                                    0, 0, 1);
    K1 = (cv::Mat_<double>(3, 3) << intrinsics1[0], 0, intrinsics1[2],
                                    0, intrinsics1[1], intrinsics1[3],
                                    0, 0, 1);
    D0.create(distortion_coefficients0.size(), 1, CV_64F);
    D1.create(distortion_coefficients1.size(), 1, CV_64F);

    for(int i = 0, n = distortion_coefficients0.size(); i < n; ++i)
        D0.at<double>(i) = distortion_coefficients0[i];

    for(int i = 0, n = distortion_coefficients1.size(); i < n; ++i)
        D1.at<double>(i) = distortion_coefficients1[i];

    cv::Mat Tc1c0 = Tbc1.inv() * Tbc0;
    cv::Mat Rc1c0, tc1c0;
    Tc1c0.rowRange(0, 3).colRange(0, 3).copyTo(Rc1c0);
    Tc1c0.col(3).rowRange(0, 3).copyTo(tc1c0);
    cv::Mat R0, R1, P0, P1; // R0 = Rcp0c0, R1 = cp1c1
    cv::stereoRectify(K0, D0, K1, D1, image_size, Rc1c0, tc1c0, R0, R1, P0, P1, cv::noArray());

    double f, cx, cy;
    double b;
    f = P0.at<double>(0, 0);
    cx = P0.at<double>(0, 2);
    cy = P0.at<double>(1, 2);
    b = -P1.at<double>(0, 3) / f;

    cv::Mat M1l, M2l, M1r, M2r;
    cv::initUndistortRectifyMap(K0, D0, R0, P0, image_size, CV_32F, M1l, M2l);
    cv::initUndistortRectifyMap(K1, D1, R1, P1, image_size, CV_32F, M1r, M2r);

    // fix entrinsics
    Eigen::Matrix4d temp_T;
    Eigen::Matrix3d temp_R;
    Eigen::Quaterniond temp_q;
    cv::cv2eigen(Tbc0, temp_T);
    temp_q = temp_T.block<3, 3>(0, 0);
    temp_q.normalized();
    Sophus::SE3d sTbc0(temp_q, temp_T.block<3, 1>(0, 3));

    cv::cv2eigen(Tbc1, temp_T);
    temp_q = temp_T.block<3, 3>(0, 0);
    temp_q.normalized();
    Sophus::SE3d sTbc1(temp_q, temp_T.block<3, 1>(0, 3));

    cv::cv2eigen(Tbi, temp_T);
    temp_q = temp_T.block<3, 3>(0, 0);
    temp_q.normalized();
    Sophus::SE3d sTbi(temp_q, temp_T.block<3, 1>(0, 3));

    cv::cv2eigen(R0, temp_R);
    Sophus::SE3d sTcp0c0;
    sTcp0c0.setRotationMatrix(temp_R);

    cv::cv2eigen(R1, temp_R);
    Sophus::SE3d sTcp1c1;
    sTcp1c1.setRotationMatrix(temp_R);

    Sophus::SE3d sTbcp0 = sTbc0 * sTcp0c0.inverse();
    Sophus::SE3d sTbcp1 = sTbc1 * sTcp1c1.inverse();

    camera = std::make_shared<SimpleStereoCam>(sTbcp0, width, height, f, cx, cy, b,
                                               M1l, M2l, M1r, M2r);

    // load imu info
    double acc_n, acc_w, gyo_n, gyo_w;
    acc_n = fs["acc_n"];
    gyo_n = fs["gyr_n"];
    acc_w = fs["acc_w"];
    gyo_w = fs["gyr_w"];

    imu_sensor = std::make_shared<ImuSensor>(sTbi, gyo_n, acc_n, gyo_w, acc_w);

    gyr_acc_cov.setZero();
    gyr_acc_cov.block<3, 3>(0, 0) = imu_sensor->gyr_noise_cov;
    gyr_acc_cov.block<3, 3>(3, 3) = imu_sensor->acc_noise_cov;

    imu_preintegration = std::make_shared<ImuPreintegration>(Eigen::Vector3d::Zero(),
                                                             Eigen::Vector3d::Zero(),
                                                             gyr_acc_cov);

    gravity_magnitude = fs["g_norm"];
    fs.release();

    sliding_window = std::make_shared<SlidingWindow>();
    front_end = std::make_shared<SimpleFrontEnd>(camera, sliding_window);
    back_end = std::make_shared<BackEnd>();
    t_backend = std::thread(&BackEnd::Process, back_end);

    // log information
    ROS_INFO_STREAM("Camera.Type  : Stereo Camera");
    ROS_INFO_STREAM("Camera.Tbs   :\n" << camera->Tbs.matrix());
    ROS_INFO_STREAM("Camera.width : " << camera->width);
    ROS_INFO_STREAM("Camera.height: " << camera->height);
    ROS_INFO_STREAM("Camera.f     : " << camera->f);
    ROS_INFO_STREAM("Camera.cx    : " << camera->cx);
    ROS_INFO_STREAM("Camera.cy    : " << camera->cy);
    ROS_INFO_STREAM("Camera.b     : " << camera->b);

    ROS_INFO_STREAM("IMU.Tbs      :\n" << imu_sensor->Tbs.matrix());
    ROS_INFO_STREAM("IMU.acc_noise: " << imu_sensor->acc_noise);
    ROS_INFO_STREAM("IMU.gyo_noise: " << imu_sensor->gyr_noise);
    ROS_INFO_STREAM("IMU.ab_noise : " << imu_sensor->acc_bias_noise);
    ROS_INFO_STREAM("IMU.gb_noise : " << imu_sensor->gyr_bias_noise);
    ROS_INFO_STREAM("gravity magnitude: " << gravity_magnitude);
}

VIOSystem::~VIOSystem() {

}

void VIOSystem::Process(const cv::Mat& img_raw_l, const cv::Mat& img_raw_r, double timestamp,
                        const std::vector<ImuData>& v_imu_data) {
    cv::Mat img_remap_l, img_remap_r;
    cv::remap(img_raw_l, img_remap_l, camera->M1l, camera->M2l, cv::INTER_LINEAR);
    cv::remap(img_raw_r, img_remap_r, camera->M1r, camera->M2r, cv::INTER_LINEAR);

    cur_frame = std::make_shared<Frame>(img_remap_l, img_remap_r, timestamp, camera);
    if(last_frame) {
        ROS_ERROR_STREAM("frame id: " << cur_frame->mFrameID);
        for(auto& it : v_imu_data) {
            ROS_ERROR_STREAM("imu t:" << it.timestamp - last_frame->mTimeStamp);
        }
    }
    else {
        cur_frame->ExtractFAST();
        sliding_window->push_mps(cur_frame->SetToKeyFrame());
    }

    last_frame = cur_frame;
    last_imu_timestamp = v_imu_data.back().timestamp;
}
