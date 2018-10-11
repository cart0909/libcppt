#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>

// From SVO SparseImgAlign
class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SparseImgAlign();
    ~SparseImgAlign();

    std::vector<cv::Mat> mRefImgPyr;
    std::vector<cv::Point2f> mRefPts;
    std::vector<cv::Mat> mCurImgPyr;
    Sophus::SE3d mTcf;

    int mMaxLevel; // coarsest pyramid level for the alignment.
    int mMinLevel; // finest pyramid level for the alignment.

    int mPatchHalfSize; // 2
    int mPatchSize;     // 4
    int mPatchArea;     // 16
};
