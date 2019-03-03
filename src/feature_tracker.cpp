#include "feature_tracker.h"

FeatureTracker::FeatureTracker(CameraPtr camera_, double clahe_parameter, int fast_threshold_,
                               int min_dist, float reproj_threshold_)
    : camera(camera_), fast_threshold(fast_threshold_), feat_min_dist(min_dist), reproj_threshold(reproj_threshold_)
{
    if(clahe_parameter > 0)
        clahe = cv::createCLAHE(clahe_parameter);
}
FeatureTracker::~FeatureTracker() {}

FeatureTracker::FramePtr FeatureTracker::InitFirstFrame(const cv::Mat& img, double timestamp) {
    next_frame_id = 0;
    next_pt_id = 0;
    FramePtr frame = InitFrame(img, timestamp);
    ExtractFAST(frame);
    last_frame = frame;
    return frame;
}

FeatureTracker::FramePtr FeatureTracker::Process(const cv::Mat& img, double timestamp) {
    FramePtr frame = InitFrame(img, timestamp);
    TrackFeatures(last_frame, frame);
    last_frame = frame;
    return frame;
}

FeatureTracker::FramePtr FeatureTracker::InitFrame(const cv::Mat& img, double timestamp) {
    ScopedTrace st("init_frame");
    FramePtr frame(new Frame);
    frame->id = next_frame_id++;
    frame->timestamp = timestamp;
    frame->img = img;
    cv::resize(img, frame->compressed_img, img.size() / 2, 0, 0, CV_INTER_NN);
    cv::cvtColor(frame->compressed_img, frame->compressed_img, CV_GRAY2BGR);
    if(clahe) {
        clahe->apply(img, frame->clahe);
        cv::buildOpticalFlowPyramid(frame->clahe, frame->img_pyr_grad, cv::Size(21, 21), 3);
    }
    else {
        cv::buildOpticalFlowPyramid(frame->img, frame->img_pyr_grad, cv::Size(21, 21), 3);
    }
    return frame;
}

void FeatureTracker::ExtractFAST(FramePtr frame) {
    ScopedTrace st("FAST");
    cv::Mat mask(camera->height, camera->width, CV_8U, cv::Scalar(255));
    std::vector<std::pair<uint, size_t>> v_count_idx(frame->pt.size());
    std::vector<uchar> v_keep(frame->pt.size(), 1);

    for(size_t i = 0, n = frame->pt.size(); i < n; ++i)
        v_count_idx[i] = std::make_pair(frame->pt_track_count[i], i);

    std::sort(v_count_idx.begin(), v_count_idx.end(), std::greater<std::pair<uint, size_t>>());

    for(size_t i = 0, n = frame->pt.size(); i < n; ++i) {
        size_t idx = v_count_idx[i].second;
        cv::Point2f& pt = frame->pt[idx];
        if(mask.at<uchar>(pt) != 0) {
            cv::circle(mask, pt, feat_min_dist, cv::Scalar(0), -1);
        }
        else {
            v_keep[idx] = 0;
        }
    }

    util::ReduceVector(frame->pt_id, v_keep);
    util::ReduceVector(frame->pt_track_count, v_keep);
    util::ReduceVector(frame->pt, v_keep);

    for(auto& it : frame->pt_track_count)
        ++it;

    std::vector<cv::KeyPoint> kps;
    cv::FAST(frame->img, kps, fast_threshold);

    for(auto& kp : kps) {
        auto& pt = kp.pt;
        if(mask.at<uchar>(pt) != 0) {
            cv::circle(mask, pt, feat_min_dist, cv::Scalar(0), -1);

            frame->pt_id.emplace_back(next_pt_id++);
            frame->pt_track_count.emplace_back(0);
            frame->pt.emplace_back(pt);
        }
    }
}

void FeatureTracker::TrackFeatures(FramePtr ref_frame, FramePtr cur_frame) {
    if(ref_frame->pt.empty())
        return;
    ScopedTrace st("opt_flow");
    // clone ref frame data to cur frame
    std::vector<cv::Point2f> ref_frame_pts = ref_frame->pt;
    cur_frame->pt_id = ref_frame->pt_id;
    cur_frame->pt_track_count = ref_frame->pt_track_count;

    // optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(ref_frame->img_pyr_grad, cur_frame->img_pyr_grad,
                             ref_frame_pts, cur_frame->pt, status, err,
                             cv::Size(21, 21), 3);

    for(int i = 0, n = cur_frame->pt.size(); i < n; ++i)
        if(status[i] && !util::InBorder(cur_frame->pt[i], camera->width, camera->height))
            status[i] = 0;

    util::ReduceVector(ref_frame_pts, status);
    util::ReduceVector(cur_frame->pt_id, status);
    util::ReduceVector(cur_frame->pt, status);
    util::ReduceVector(cur_frame->pt_track_count, status);

    // check fundamental matrix
    if(cur_frame->pt.size() > 8) {
        size_t N = ref_frame_pts.size();
        float f = camera->f();
        float cx = camera->width / 2;
        float cy = camera->height / 2;
        std::vector<cv::Point2f> ref_un_pts(N), cur_un_pts(N);

        for(size_t i = 0; i < N; ++i) {
            Eigen::Vector3d P;
            camera->BackProject(Eigen::Vector2d(ref_frame_pts[i].x, ref_frame_pts[i].y), P);
            ref_un_pts[i].x = f * P(0) + cx;
            ref_un_pts[i].y = f * P(1) + cy;

            camera->BackProject(Eigen::Vector2d(cur_frame->pt[i].x, cur_frame->pt[i].y), P);
            cur_un_pts[i].x = f * P(0) + cx;
            cur_un_pts[i].y = f * P(1) + cy;
        }

        status.clear();
        cv::findFundamentalMat(ref_un_pts, cur_un_pts, cv::FM_RANSAC, reproj_threshold, 0.99, status);
        util::ReduceVector(ref_frame_pts, status);
        util::ReduceVector(cur_frame->pt_id, status);
        util::ReduceVector(cur_frame->pt, status);
        util::ReduceVector(cur_frame->pt_track_count, status);
    }

    for(auto& it : cur_frame->pt_track_count) {
        ++it;
    }
}
