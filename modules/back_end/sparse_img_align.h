#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "camera_model/pinhole_camera.h"
#include "ceres/intensity_factor.h"

// From SVO SparseImgAlign
class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    using ImagePyr = std::vector<cv::Mat>;
    using VecVector2d = std::vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>>;
    using VecVector3d = std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>;
    SparseImgAlign(int max_iter = 10, int max_level = 2, int min_level = 0)
        : mMaxLevel(max_level), mMinLevel(min_level), mMaxIter(max_iter) {}
    ~SparseImgAlign() {}

    void SetState(const ImagePyr& ref_img_pyr, const ImagePyr& cur_img_pyr,
                  const VecVector2d& ref_pts, const VecVector3d& ref_mps,
                  const Sophus::SE3d& predict_Tcr);
    bool Run(Sophus::SE3d& Tcr);

    ImagePyr mvRefImgPyr;
    ImagePyr mvCurImgPyr;
    VecVector2d mvRefPts;
    VecVector3d mvRefMps; // x3Dw
    Sophus::SE3d mTrw;
    Sophus::SE3d mTcr;

    int mMaxLevel; // coarsest pyramid level for the alignment.
    int mMinLevel; // finest pyramid level for the alignment.
    int mCurLevel;

    int mMaxIter; // 10?
    PinholeCameraPtr mpCamera;
};

using SparseImgAlignPtr = std::shared_ptr<SparseImgAlign>;
using SparseImgAlignConstPtr = std::shared_ptr<const SparseImgAlign>;
