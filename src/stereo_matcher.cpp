#include "stereo_matcher.h"
#include <glog/logging.h>

StereoMatcher::StereoMatcher(CameraPtr cam_l_, CameraPtr cam_r_, Eigen::Vector3d prl_, Sophus::SO3d qrl_)
    : cam_l(cam_l_), cam_r(cam_r_), prl(prl_), qrl(qrl_)
{
    clahe = cv::createCLAHE(3.0f);
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

    return frame;
}

StereoMatcher::FramePtr StereoMatcher::InitFrame(const cv::Mat& img_r) {
    ScopedTrace st("init_rFrame");
    FramePtr frame(new Frame);
    frame->img_r = img_r;
    clahe->apply(frame->img_r, frame->clahe_r);
    cv::buildOpticalFlowPyramid(frame->clahe_r, frame->img_pyr_grad_r, cv::Size(21, 21), 3);
    return frame;
}
