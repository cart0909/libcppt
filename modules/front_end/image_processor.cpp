#include "image_processor.h"
#include "basic_datatype/tic_toc.h"

bool InBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

template<class T>
void ReduceVector(std::vector<T> &v, const std::vector<uchar>& status) {
    int j = 0;
    for(int i = 0, n = v.size(); i < n; ++i) {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

ImageProcessor::ImageProcessor() : mNextPtId(0) {}
ImageProcessor::~ImageProcessor() {}

void ImageProcessor::ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp) {
    cv::buildOpticalFlowPyramid(imLeft, mvCurImagePyramid, cv::Size(21, 21), 3);

    if(!mvLastPts.empty()) {
        std::vector<uchar> status;
        std::vector<float> err;
        // optical flow
        cv::calcOpticalFlowPyrLK(mvLastImagePyramid, mvCurImagePyramid, mvLastPts, mvCurPts, status,
                                 err, cv::Size(21, 21), 3);

        for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
            if(status[i] && !InBorder(mvCurPts[i]))
                status[i] = 0;
        }

        ReduceVector(mvLastPts, status);
        ReduceVector(mvCurPts, status);
        ReduceVector(mvIds, status);
        ReduceVector(mvTrackCnt, status);
        // check epipolar constrain
        RemoveOutlierFromF();
    }

    // set mask avoiding new feature points close to old points
    SetMask();

    // detect new
    cv::goodFeaturesToTrack(mvCurImagePyramid[0], mvNewPts, MAX_CNT - mvLastPts.size(), 0.01, MIN_DIST);

    // register new point
    for(int i = 0, n = mvNewPts.size(); i < n; ++i) {
        mvCurPts.emplace_back(mvNewPts[i]);
        mvIds.emplace_back(mNextPtId++);
        mvTrackCnt.emplace_back(0);
    }

    // stereo matching

    // check epipolar constrain

    // update point
    mvLastPts = mvCurPts;
    mvLastImagePyramid = mvCurImagePyramid;

    //    cv::Mat result;
    //    cv::cvtColor(mvCurImagePyramid[0], result, CV_GRAY2BGR);

    //    for(auto& it : mvCurPts) {
    //        cv::circle(result, it, 3, cv::Scalar(0, 255, 0), -1);
    //    }

    //    cv::imshow("result", result);
    //    cv::waitKey(1);

}

void ImageProcessor::ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp,
                                const std::vector<ImuData>& imu_data) {

}

void ImageProcessor::SetMask() {
    if(mvCurPts.empty())
        return;

    mMask = cv::Mat(ROW, COL, CV_8U, cv::Scalar(255));

    // Prefer to keep features that are tracked for long time
    std::vector<std::pair<int, std::pair<cv::Point2f, int>>> v_cnt_pts_id;

    for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
        v_cnt_pts_id.emplace_back(mvTrackCnt[i], std::make_pair(mvCurPts[i], mvIds[i]));
    }

    std::sort(v_cnt_pts_id.begin(), v_cnt_pts_id.end(),
           [](const std::pair<int, std::pair<cv::Point2f, int>>& lhs,
              const std::pair<int, std::pair<cv::Point2f, int>>& rhs) {
        return lhs.first > rhs.first;
    });

    mvCurPts.clear();
    mvIds.clear();
    mvTrackCnt.clear();

    for(auto& it : v_cnt_pts_id) {
        if(mMask.at<uchar>(it.second.first) == 255) {
            mvCurPts.emplace_back(it.second.first);
            mvIds.emplace_back(it.second.second);
            mvTrackCnt.emplace_back(it.first);
            cv::circle(mMask, it.second.first, MIN_DIST, 0, -1);
        }
    }
}

void ImageProcessor::RemoveOutlierFromF() {
    if(mvCurPts.size() > 8) {
        std::vector<cv::Point2f> un_last_pts(mvLastPts.size()), un_cur_pts(mvCurPts.size());
        auto left_cam = mpStereoCam->mpCamera[0];
        double fx = left_cam->fx;
        double fy = left_cam->fy;
        double cx = left_cam->cx;
        double cy = left_cam->cy;
        for(int i = 0, n = mvLastPts.size(); i < n; ++i) {
            Eigen::Vector3d tmp_p;
            left_cam->BackProject(Eigen::Vector2d(mvLastPts[i].x, mvLastPts[i].y), tmp_p);
            tmp_p.x() = fx * tmp_p.x() / tmp_p.z() + cx;
            tmp_p.y() = fy * tmp_p.y() / tmp_p.z() + cy;
            un_last_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());

            left_cam->BackProject(Eigen::Vector2d(mvCurPts[i].x, mvCurPts[i].y), tmp_p);
            tmp_p.x() = fx * tmp_p.x() / tmp_p.z() + cx;
            tmp_p.y() = fy * tmp_p.y() / tmp_p.z() + cy;
            un_cur_pts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        std::vector<uchar> status;
        cv::findFundamentalMat(un_last_pts, un_cur_pts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        ReduceVector(mvLastPts, status);
        ReduceVector(mvCurPts, status);
        ReduceVector(mvIds, status);
        ReduceVector(mvTrackCnt, status);
    }
}
