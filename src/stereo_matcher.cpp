#include "stereo_matcher.h"
#include <glog/logging.h>

StereoMatcher::StereoMatcher(CameraPtr cam_l_, CameraPtr cam_r_,
                             const Eigen::Vector3d& prl_, const Sophus::SO3d& qrl_,
                             double clahe_parameter, float dist_epipolar_threshold_)
    : cam_l(cam_l_), cam_r(cam_r_), prl(prl_), qrl(qrl_), dist_epipolar_threshold(dist_epipolar_threshold_),
      know_camera_extrinsic(true)
{
    if(clahe_parameter > 0)
        clahe = cv::createCLAHE(clahe_parameter);
}

StereoMatcher::StereoMatcher(CameraPtr cam_l_, CameraPtr cam_r_, double clahe_parameter,
                             float dist_epipolar_threshold_)
    : cam_l(cam_l_), cam_r(cam_r_), dist_epipolar_threshold(dist_epipolar_threshold_),
      know_camera_extrinsic(false)
{
    if(clahe_parameter > 0)
        clahe = cv::createCLAHE(clahe_parameter);
}

StereoMatcher::~StereoMatcher() {

}

StereoMatcher::FramePtr StereoMatcher::Process(FeatureTracker::FrameConstPtr feat_frame, const cv::Mat& img_r) {
    if(img_r.empty() || img_r.type() != CV_8U) {
        throw std::runtime_error("img_r empty or type error!");
    }

    FramePtr frame = InitFrame(img_r);

    if(feat_frame->pt.empty())
        return frame;
    ScopedTrace st("stereo_match");
    // optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(feat_frame->img_pyr_grad, frame->img_pyr_grad_r, feat_frame->pt, frame->pt_r,
                             status, err, cv::Size(21, 21), 3);

    for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
        if(status[i] && !util::InBorder(frame->pt_r[i], cam_l->width, cam_l->height)) {
            status[i] = 0;
        }
    }

    if(know_camera_extrinsic) {
        FundaMatCheck(feat_frame, frame, status);
    }
    else {
        LeftRightCheck(feat_frame, frame, status);
    }

    // debug
#if 0
    cv::Mat left_img = feat_frame->compressed_img;
    cv::Mat right_img = frame->compressed_img_r;
    cv::Mat result;
    cv::hconcat(left_img, right_img, result);

    for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
        cv::Point2f pt_l = feat_frame->pt[i] / 2;
        cv::circle(result, pt_l, 2, cv::Scalar(0, 255, 0), -1);
        if(frame->pt_r[i].x != -1.0) {
            cv::Point2f pt_r = frame->pt_r[i] / 2;
            pt_r.x += left_img.cols;
            cv::circle(result, pt_r, 2, cv::Scalar(0, 0, 255), -1);
        }
    }

    cv::imshow("stereo_matching", result);
    cv::waitKey(1);
#endif
    return frame;
}

StereoMatcher::FramePtr StereoMatcher::InitFrame(const cv::Mat& img_r) {
    ScopedTrace st("init_rFrame");
    FramePtr frame(new Frame);
    frame->img_r = img_r;
    cv::resize(frame->img_r, frame->compressed_img_r, frame->img_r.size() / 2, 0, 0, CV_INTER_NN);
    cv::cvtColor(frame->compressed_img_r, frame->compressed_img_r, CV_GRAY2BGR);

    if(clahe) {
        clahe->apply(frame->img_r, frame->clahe_r);
        cv::buildOpticalFlowPyramid(frame->clahe_r, frame->img_pyr_grad_r, cv::Size(21, 21), 3);
    }
    else {
        cv::buildOpticalFlowPyramid(frame->img_r, frame->img_pyr_grad_r, cv::Size(21, 21), 3);
    }
    return frame;
}

void StereoMatcher::FundaMatCheck(FeatureTracker::FrameConstPtr feat_frame, FramePtr frame,
                                  std::vector<uchar>& status)
{
    // compute fundamental matrix
    Eigen::Matrix3d E = qrl.matrix() * Sophus::SO3d::hat(prl);
    E /= E(2, 2);

    double f = cam_l->f();
    double cx = static_cast<double>(cam_l->width) / 2;
    double cy = static_cast<double>(cam_l->height) / 2;
    Eigen::Matrix3d K;
    K << f, 0, cx,
         0, f, cy,
         0, 0,  1;
    Eigen::Matrix3d F = K.transpose().inverse() * E * K.inverse();
    F /= F(2, 2);

    // check epipolar constrain
    for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
        if(status[i] == 0)
            continue;

        Eigen::Vector3d Pl, Pr;
        Eigen::Vector3d xl, xr;
        cam_l->BackProject(Eigen::Vector2d(feat_frame->pt[i].x, feat_frame->pt[i].y), Pl);
        cam_r->BackProject(Eigen::Vector2d(frame->pt_r[i].x, frame->pt_r[i].y), Pr);
        xl << f * Pl(0) + cx, f * Pl(1) + cy, 1;
        xr << f * Pr(0) + cx, f * Pr(1) + cy, 1;

        Eigen::RowVector3d lt = xr.transpose() * F;
        double dist_epipolar = std::abs(lt * xl) / lt.leftCols(2).norm();

        if(dist_epipolar >= dist_epipolar_threshold) {
            status[i] = 0;
        }
    }

    for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
        if(status[i] == 0) {
            frame->pt_r[i] = cv::Point2f(-1, -1);
        }
    }
}

void StereoMatcher::LeftRightCheck(FeatureTracker::FrameConstPtr feat_frame, FramePtr frame,
                                   std::vector<uchar>& status)
{
    // left right consistency check
    std::vector<size_t> inlier_index;
    for(int i = 0, n = feat_frame->pt.size(); i < n; ++i) {
        if(status[i])
            inlier_index.emplace_back(i);
        else
            frame->pt_r[i] = cv::Point2f(-1, -1);
    }

    std::vector<cv::Point2f> feat_frame_pt = feat_frame->pt;
    std::vector<cv::Point2f> frame_pt = frame->pt_r;

    util::ReduceVector(feat_frame_pt, status);
    util::ReduceVector(frame_pt, status);

    if(!frame_pt.empty()) {
        status.clear();
        std::vector<float> err;
        std::vector<cv::Point2f> feat_frame_pt_calc = feat_frame_pt;
        cv::calcOpticalFlowPyrLK(frame->img_pyr_grad_r, feat_frame->img_pyr_grad,
                                 frame_pt, feat_frame_pt_calc,
                                 status, err, cv::Size(21, 21), 3,
                                 cv::TermCriteria(cv::TermCriteria::COUNT+ cv::TermCriteria::EPS, 30, 0.01),
                                 cv::OPTFLOW_USE_INITIAL_FLOW);

        for(int i = 0, n = frame_pt.size(); i < n; ++i) {
            if(status[i] && !util::InBorder(feat_frame_pt_calc[i], cam_l->width, cam_l->height)) {
                status[i] = 0;
            }

            if(status[i]) {
                cv::Point2f delta = feat_frame_pt_calc[i] - feat_frame_pt[i];
                if(delta.dot(delta) >= dist_epipolar_threshold * dist_epipolar_threshold) {
                    frame->pt_r[inlier_index[i]] = cv::Point2f(-1, -1);
                }
            }
            else {
                frame->pt_r[inlier_index[i]] = cv::Point2f(-1, -1);
            }
        }
    }
}
