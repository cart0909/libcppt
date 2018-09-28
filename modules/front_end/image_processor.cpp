#include "image_processor.h"

bool InBorder(const cv::Point2f &pt)
{
    const int BORDER_SIZE = 1;
    int img_x = cvRound(pt.x);
    int img_y = cvRound(pt.y);
    return BORDER_SIZE <= img_x && img_x < COL - BORDER_SIZE && BORDER_SIZE <= img_y && img_y < ROW - BORDER_SIZE;
}

template<class T>
void ReduceVector(std::vector<T> &v, std::vector<uchar> status) {
    int j = 0;
    for(int i = 0, n = v.size(); i < n; ++i) {
        if(status[i])
            v[j++] = v[i];
    }
    v.resize(j);
}

ImageProcessor::ImageProcessor() {}
ImageProcessor::~ImageProcessor() {}

void ImageProcessor::ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp) {
    cv::buildOpticalFlowPyramid(imLeft, mvCurImagePyramid, cv::Size(21, 21), 3);

    if(!mvLastPts.empty()) {
        std::vector<uchar> status;
        std::vector<float> err;
        // optical flow
        cv::calcOpticalFlowPyrLK(mvLastImagePyramid, mvCurImagePyramid, mvLastPts, mvCurPts, status,
                                 err, cv::Size(21, 21), 3);
        // check epipolar constrain
    }

    // set mask avoiding new feature points close to old points
    SetMask();

    // detect new
    cv::goodFeaturesToTrack(mvCurImagePyramid[0], mvNewPts, MAX_CNT - mvLastPts.size(), 0.01, MIN_DIST);

    // stereo matching

    // check epipolar constrain

    // register point

//    cv::Mat result;
//    cv::cvtColor(mvCurImagePyramid[0], result, CV_GRAY2BGR);

//    for(auto& it : mvCurPts) {
//        cv::circle(result, it, 3, cv::Scalar(0, 255, 0), -1);
//    }

//    cv::imshow("result", result);
//    cv::waitKey(1);

    mvLastPts = mvCurPts;
    mvLastImagePyramid = mvCurImagePyramid;
}

void ImageProcessor::ReadStereo(const cv::Mat& imLeft, const cv::Mat& imRight, double timestamp,
                                const std::vector<ImuData>& imu_data) {

}

void ImageProcessor::SetMask() {
    mMask = cv::Mat(ROW, COL, CV_8U, cv::Scalar(255));

    if(!mvCurPts.empty()) {

    }
}
