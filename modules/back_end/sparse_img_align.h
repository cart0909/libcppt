#pragma once
#include <vector>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "camera_model/pinhole_camera.h"

// From SVO SparseImgAlign
class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SparseImgAlign();
    ~SparseImgAlign();

    void Run();
    bool Solve();
    void Update();
    void Optimize();
    void ComputeReferencePatchAndJacobian();

    std::vector<cv::Mat> mvRefImgPyr;
    std::vector<cv::Mat> mvCurImgPyr;
    std::vector<Eigen::Vector2d> mvRefPts;
    std::vector<Eigen::Vector3d> mvRefMps; // x3Dw
    Sophus::SE3d mTrw;
    Sophus::SE3d mTcr;

    int mMaxLevel; // coarsest pyramid level for the alignment.
    int mMinLevel; // finest pyramid level for the alignment.
    int mCurLevel;

    int mPatchHalfSize; // 2
    int mPatchSize;     // 4
    int mPatchArea;     // 16

    int mMaxIter; // 10?

    cv::Mat mRefPatch;
    Eigen::Matrix<double, 6, Eigen::Dynamic> mJpatch_pt;
    std::vector<bool> mvIsVisible;
    Eigen::Matrix<double, 6, 6> mH;
    Eigen::Matrix<double, 6, 1> mb, mdelx;
    PinholeCameraPtr mpCamera;
};
