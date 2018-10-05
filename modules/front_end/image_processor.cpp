#include "image_processor.h"
#include "basic_datatype/tic_toc.h"
#include <ros/ros.h>

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
    TicToc tic;
    mCurImage = imLeft;
    mCurImageR = imRight;

    if(!mvLastPts.empty()) {
        std::vector<uchar> status;
        std::vector<float> err;
        // optical flow
        cv::calcOpticalFlowPyrLK(mLastImage, mCurImage, mvLastPts, mvCurPts, status,
                                 err, cv::Size(21, 21), 3);

        for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
            if(status[i] && !InBorder(mvCurPts[i]))
                status[i] = 0;
        }

        ReduceVector(mvLastPts, status);
        ReduceVector(mvLastUnPts, status);
        ReduceVector(mvCurPts, status);
        ReduceVector(mvIds, status);
        ReduceVector(mvTrackCnt, status);
        // check epipolar constrain
        RemoveOutlierFromF();

        // add track count
        for(int i = 0, n = mvCurPts.size(); i < n; ++i)
            mvTrackCnt[i] += 1;
    }

    // set mask avoiding new feature points close to old points
    SetMask();

    // detect new
    if(mvCurPts.size() < MAX_CNT) {
        cv::goodFeaturesToTrack(mCurImage, mvNewPts, MAX_CNT - mvCurPts.size(),
                                0.01, MIN_DIST, mMask);
        auto left_cam = mpStereoCam->mpCamera[0];
        double fx = left_cam->fx;
        double fy = left_cam->fy;
        double cx = left_cam->cx;
        double cy = left_cam->cy;
        // register new point
        for(int i = 0, n = mvNewPts.size(); i < n; ++i) {
            mvCurPts.emplace_back(mvNewPts[i]);
            mvIds.emplace_back(mNextPtId++);
            mvTrackCnt.emplace_back(0);
            Eigen::Vector3d tmp_p;
            left_cam->BackProject(Eigen::Vector2d(mvNewPts[i].x, mvNewPts[i].y), tmp_p);
            tmp_p.x() = fx * tmp_p.x() / tmp_p.z() + cx;
            tmp_p.y() = fy * tmp_p.y() / tmp_p.z() + cy;
            mvCurUnPts.emplace_back(tmp_p.x(), tmp_p.y());
        }
    }

    // stereo matching
    {
        std::vector<float> err;
        // optical flow
        cv::calcOpticalFlowPyrLK(mCurImage, mCurImageR, mvCurPts, mvCurPtsR, mvIsStereo,
                                 err, cv::Size(21, 21), 3);
        mNumStereo = 0;
        for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
            if(mvIsStereo[i]) {
                if(!InBorder(mvCurPtsR[i]))
                    mvIsStereo[i] = 0;
                else
                    ++mNumStereo;
            }
        }
        // check epipolar constrain
        CheckStereoConstrain();
    }

    // update states
    mvLastPts = mvCurPts;
    mvLastUnPts = mvCurUnPts;
    mLastImage = mCurImage;
    mLastImageR = mCurImageR;

    ROS_DEBUG_STREAM("ImageProcess cost " << tic.toc());

    cv::Mat result, result_r;
    cv::cvtColor(mCurImage, result, CV_GRAY2BGR);
    cv::cvtColor(mCurImageR, result_r, CV_GRAY2BGR);
    const int max_count = 25;
    const float each_step = 255.0 / max_count;
    for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
        if(mvTrackCnt[i] < 0) {
            cv::circle(result, mvCurPts[i], 3, cv::Scalar(255, 0, 0), -1);
            if(mvIsStereo[i])
                cv::circle(result_r, mvCurPtsR[i], 3, cv::Scalar(255, 0, 0), -1);
        }
        else if(mvTrackCnt[i] >= max_count) {
            cv::circle(result, mvCurPts[i], 3, cv::Scalar(0, 0, 255), -1);
            if(mvIsStereo[i])
                cv::circle(result_r, mvCurPtsR[i], 3, cv::Scalar(0, 0, 255), -1);
        }
        else {
            cv::circle(result, mvCurPts[i], 3, cv::Scalar(255 - each_step * mvTrackCnt[i], 0,
                                                          each_step * mvTrackCnt[i]), -1);
            if(mvIsStereo[i])
                cv::circle(result_r, mvCurPtsR[i], 3, cv::Scalar(255 - each_step * mvTrackCnt[i], 0,
                                                                 each_step * mvTrackCnt[i]), -1);
        }
    }

    cv::hconcat(result, result_r, result);
    cv::imshow("result", result);
    cv::waitKey(1);
}

void ImageProcessor::ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp,
                                const std::vector<ImuData>& imu_data) {

}

void ImageProcessor::SetMask() {
    if(mvCurPts.empty())
        return;

    mMask = cv::Mat(ROW, COL, CV_8U, cv::Scalar(255));

    // Prefer to keep features that are tracked for long time
    std::vector<std::tuple<int, cv::Point2d, cv::Point2d, int>> v_cnt_pts_unpts_id;

    for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
        v_cnt_pts_unpts_id.emplace_back(mvTrackCnt[i], mvCurPts[i], mvCurUnPts[i], mvIds[i]);
    }

    std::sort(v_cnt_pts_unpts_id.begin(), v_cnt_pts_unpts_id.end(),
           [](const std::tuple<int, cv::Point2d, cv::Point2d, int>& lhs,
              const std::tuple<int, cv::Point2d, cv::Point2d, int>& rhs) {
        return std::get<0>(lhs) > std::get<0>(rhs);
    });

    mvCurPts.clear();
    mvCurUnPts.clear();
    mvIds.clear();
    mvTrackCnt.clear();

    for(auto& it : v_cnt_pts_unpts_id) {
        if(mMask.at<uchar>(std::get<1>(it)) == 255) {
            mvCurPts.emplace_back(std::get<1>(it));
            mvCurUnPts.emplace_back(std::get<2>(it));
            mvIds.emplace_back(std::get<3>(it));
            mvTrackCnt.emplace_back(std::get<0>(it));
            cv::circle(mMask, std::get<1>(it), MIN_DIST, 0, -1);
        }
    }
}

void ImageProcessor::RemoveOutlierFromF() {
    if(mvCurPts.size() > 8) {
        mvCurUnPts.resize(mvCurPts.size());
        auto left_cam = mpStereoCam->mpCamera[0];
        double fx = left_cam->fx;
        double fy = left_cam->fy;
        double cx = left_cam->cx;
        double cy = left_cam->cy;
        for(int i = 0, n = mvLastPts.size(); i < n; ++i) {
            Eigen::Vector3d tmp_p;
            left_cam->BackProject(Eigen::Vector2d(mvCurPts[i].x, mvCurPts[i].y), tmp_p);
            tmp_p.x() = fx * tmp_p.x() / tmp_p.z() + cx;
            tmp_p.y() = fy * tmp_p.y() / tmp_p.z() + cy;
            mvCurUnPts[i] = cv::Point2f(tmp_p.x(), tmp_p.y());
        }

        std::vector<uchar> status;
        cv::findFundamentalMat(mvLastUnPts, mvCurUnPts, cv::FM_RANSAC, F_THRESHOLD, 0.99, status);
        ReduceVector(mvCurPts, status);
        ReduceVector(mvCurUnPts, status);
        ReduceVector(mvIds, status);
        ReduceVector(mvTrackCnt, status);
    }
}

void ImageProcessor::CheckStereoConstrain() {
    auto right_cam = mpStereoCam->mpCamera[1];
    double fx = right_cam->fx;
    double fy = right_cam->fy;
    double cx = right_cam->cx;
    double cy = right_cam->cy;
    for(int i = 0, n = mvCurPts.size(); i < n; ++i) {
        if(mvIsStereo[i]) {
            Eigen::Vector3d xr, xl;
            right_cam->BackProject(Eigen::Vector2d(mvCurPtsR[i].x, mvCurPtsR[i].y), xr);
            xr /= xr.z();
            xr.x() = fx * xr.x() + cx;
            xr.y() = fy * xr.y() + cy;

            xl << mvCurUnPts[i].x, mvCurUnPts[i].y, 1;

            Eigen::Matrix<double, 1, 3> lt = xr.transpose() * mpStereoCam->mF;
            double dist_epipolar = std::abs(lt * xl) / lt.block<1,2>(0, 0).norm();

            if(dist_epipolar >= F_THRESHOLD) {
                mvIsStereo[i] = 0;
            }
        }
    }
}
