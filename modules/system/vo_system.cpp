#include "vo_system.h"
#include <opencv2/core/eigen.hpp>

VOSystem::VOSystem(const std::string& config_file) {
    cv::FileStorage fs(config_file, cv::FileStorage::READ);

    cv::Size image_size;
    image_size.height = fs["image_height"];
    image_size.width = fs["image_width"];

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
    cv::cv2eigen(Tbc0, temp_T);
    Sophus::SE3d sTbc0(temp_T);

    cv::cv2eigen(Tbc1, temp_T);
    Sophus::SE3d sTbc1(temp_T);

    cv::cv2eigen(Tbi, temp_T);
    Sophus::SE3d sTbi(temp_T);

    cv::cv2eigen(R0, temp_R);
    Sophus::SE3d sTcp0c0;
    sTcp0c0.setRotationMatrix(temp_R);

    cv::cv2eigen(R1, temp_R);
    Sophus::SE3d sTcp1c1;
    sTcp1c1.setRotationMatrix(temp_R);

    Sophus::SE3d sTbcp0 = sTbc0 * sTcp0c0.inverse();
    Sophus::SE3d sTbcp1 = sTbc1 * sTcp1c1.inverse();

    mpStereoCam = std::make_shared<SimpleStereoCam>(sTbcp0, image_size.width, image_size.height,
                                                    f, cx, cy, b, M1l, M2l, M1r, M2r);
    mpFrontEnd = std::make_shared<SimpleFrontEnd>(mpStereoCam);
    mpBackEnd = std::make_shared<ISAM2BackEnd>(mpStereoCam);

    fs.release();
}

VOSystem::~VOSystem() {

}

void VOSystem::Process(const cv::Mat& img_raw_l, const cv::Mat& img_raw_r) {
    cv::Mat img_remap_l, img_remap_r;
    cv::remap(img_raw_l, img_remap_l, mpStereoCam->M1l, mpStereoCam->M2l, cv::INTER_LINEAR);
    cv::remap(img_raw_r, img_remap_r, mpStereoCam->M1r, mpStereoCam->M2r, cv::INTER_LINEAR);

    FramePtr frame(new Frame(img_remap_l, img_remap_r));
    if(mpLastFrame) {
        mpFrontEnd->TrackFeaturesByOpticalFlow(mpLastFrame, frame);
        if(frame->CheckKeyFrame()) {
            frame->SetToKeyFrame();
            mpFrontEnd->ExtractFeatures(frame);
        }
        mpFrontEnd->SparseStereoMatching(frame);
    }
    else {
        frame->SetToKeyFrame();
        mpFrontEnd->ExtractFeatures(frame);
        mpFrontEnd->SparseStereoMatching(frame);
    }
    mpLastFrame = frame;

    cv::Mat result, result_r;
    cv::cvtColor(frame->mImgL, result, CV_GRAY2BGR);
    cv::cvtColor(frame->mImgR, result_r, CV_GRAY2BGR);
    for(int i = 0, n = frame->mv_uv.size(); i < n; ++i) {
        auto& pt = frame->mv_uv[i];
        auto& ur = frame->mv_ur[i];
        cv::circle(result, pt, 2, cv::Scalar(0, 255, 0), -1);
        if(ur != -1)
            cv::circle(result_r, cv::Point(ur, pt.y), 2, cv::Scalar(0, 255, 0), -1);
    }
    cv::hconcat(result, result_r, result);
    cv::imshow("a", result);
    cv::waitKey(1);
}
