#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "basic_datatype/util_datatype.h"
#include "camera_model/simple_stereo_camera.h"
#include "basic_datatype/frame.h"

class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SparseImgAlign(const SimpleStereoCamPtr& camera);
    ~SparseImgAlign();

    // return estimate_Tcr
    Sophus::SE3d Run(const FramePtr& cur_frame, const FramePtr& ref_frame,
                     Sophus::SE3d init_Tcr = Sophus::SE3d());

private:
    static const int mPatchSize, mPatchHalfSize, mPatchArea;

    void Solve();
    void PrecomputeCache();
    double ComputeResidual();
    void SolveHxbUpdate();

    SimpleStereoCamPtr mpCamera;
    std::vector<cv::Point2f> mvRef_uv;
    VecVector3d mv_x3Dr;
    ImagePyr mImgPyrRef, mImgPyrCur;
    int mLevel;
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> mJacobianCache;
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic> mRefPatchCache;
    std::vector<bool> mvVisible;
    int mMaxIter;

    Sophus::SE3d mTcr;
    Eigen::Matrix<double, 6, 6> mH;
    Eigen::Matrix<double, 6, 1> mb;
};

SMART_PTR(SparseImgAlign)
