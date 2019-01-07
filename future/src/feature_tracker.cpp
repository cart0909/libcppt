#include "feature_tracker.h"

FeatureTracker::FeatureTracker(CameraPtr camera_) : camera(camera_) {
    clahe = cv::createCLAHE(3.0);
}
FeatureTracker::~FeatureTracker() {}

FeatureTracker::FramePtr FeatureTracker::InitFirstFrame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp) {
    next_frame_id = 0;
    next_pt_id = 0;
    FramePtr frame = InitFrame(img_l, img_r, timestamp);
    frame->b_keyframe = true;
    ExtractFAST(frame);
    last_frame = frame;
    return frame;
}

FeatureTracker::FramePtr FeatureTracker::Process(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp) {
    FramePtr frame = InitFrame(img_l, img_r, timestamp);
    TrackFeatures(last_frame, frame);
    if(frame->id % 2 == 0) {
        ExtractFAST(frame);
    }
    last_frame = frame;
    return frame;
}

FeatureTracker::FramePtr FeatureTracker::InitFrame(const cv::Mat& img_l, const cv::Mat& img_r, double timestamp) {
    if(img_l.empty() || img_l.type() != CV_8U) {
        throw cppt_exception("left img type error or empty");
    }

    FramePtr frame(new Frame);
    frame->id = next_frame_id++;
    frame->timestamp = timestamp;

    frame->img_l = img_l;
    clahe->apply(img_l, frame->clahe_l);
    cv::buildOpticalFlowPyramid(frame->clahe_l, frame->img_pyr_grad_l, cv::Size(21, 21), 3);

    if(!img_r.empty() && img_r.type() == CV_8U) {
        frame->img_r = img_r;
        clahe->apply(img_r, frame->clahe_r);
        cv::buildOpticalFlowPyramid(frame->clahe_r, frame->img_pyr_grad_r, cv::Size(21, 21), 3);
    }
    return frame;
}

void FeatureTracker::ExtractFAST(FramePtr frame) {
    cv::Mat mask(camera->height, camera->width, CV_8U, cv::Scalar(255));
    std::vector<std::pair<uint, size_t>> v_count_idx(frame->pt_l.size());
    std::vector<uchar> v_keep(frame->pt_l.size(), 1);

    for(size_t i = 0, n = frame->pt_l.size(); i < n; ++i)
        v_count_idx[i] = std::make_pair(frame->pt_track_count[i], i);

    std::sort(v_count_idx.begin(), v_count_idx.end(), std::greater<std::pair<uint, size_t>>());

    for(size_t i = 0, n = frame->pt_l.size(); i < n; ++i) {
        size_t idx = v_count_idx[i].second;
        cv::Point2f& pt = frame->pt_l[idx];
        if(mask.at<uchar>(pt) != 0) {
            cv::circle(mask, pt, feat_min_dist, cv::Scalar(0), -1);
        }
        else {
            v_keep[idx] = 0;
        }
    }

    util::ReduceVector(frame->pt_id, v_keep);
    util::ReduceVector(frame->pt_track_count, v_keep);
    util::ReduceVector(frame->pt_l, v_keep);

    for(auto& it : frame->pt_track_count)
        ++it;

    std::vector<cv::KeyPoint> kps;
    cv::FAST(frame->img_l, kps, fast_threshold);

    for(auto& kp : kps) {
        auto& pt = kp.pt;
        if(mask.at<uchar>(pt) != 0) {
            cv::circle(mask, pt, feat_min_dist, cv::Scalar(0), -1);

            frame->pt_id.emplace_back(next_pt_id++);
            frame->pt_track_count.emplace_back(0);
            frame->pt_l.emplace_back(pt);
        }
    }
}

void FeatureTracker::TrackFeatures(FramePtr ref_frame, FramePtr cur_frame) {
    if(ref_frame->pt_l.empty())
        return;

    // clone ref frame data to cur frame
    std::vector<cv::Point2f> ref_frame_pts = ref_frame->pt_l;
    cur_frame->pt_id = ref_frame->pt_id;
    cur_frame->pt_track_count = ref_frame->pt_track_count;

    // optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    cv::calcOpticalFlowPyrLK(ref_frame->img_pyr_grad_l, cur_frame->img_pyr_grad_l,
                             ref_frame_pts, cur_frame->pt_l, status, err,
                             cv::Size(21, 21), 3);

    for(int i = 0, n = cur_frame->pt_l.size(); i < n; ++i)
        if(status[i] && !util::InBorder(cur_frame->pt_l[i], camera->width, camera->height))
            status[i] = 0;

    util::ReduceVector(ref_frame_pts, status);
    util::ReduceVector(cur_frame->pt_id, status);
    util::ReduceVector(cur_frame->pt_l, status);
    util::ReduceVector(cur_frame->pt_track_count, status);

    // check fundamental matrix
    if(cur_frame->pt_l.size() > 8) {
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

            camera->BackProject(Eigen::Vector2d(cur_frame->pt_l[i].x, cur_frame->pt_l[i].y), P);
            cur_un_pts[i].x = f * P(0) + cx;
            cur_un_pts[i].y = f * P(1) + cy;
        }

        status.clear();
        cv::findFundamentalMat(ref_un_pts, cur_un_pts, cv::FM_RANSAC, 3, 0.99, status);
        util::ReduceVector(ref_frame_pts, status);
        util::ReduceVector(cur_frame->pt_id, status);
        util::ReduceVector(cur_frame->pt_l, status);
        util::ReduceVector(cur_frame->pt_track_count, status);
    }

    for(auto& it : cur_frame->pt_track_count) {
        ++it;
    }
}
