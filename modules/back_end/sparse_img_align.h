#pragma once
#include <vector>
#include <memory>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <sophus/se3.hpp>
#include "front_end/utility.h"
#include "camera_model/pinhole_camera.h"
#include "sparse_img_align_impl.h"

// From SVO SparseImgAlign
class SparseImgAlign {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SparseImgAlign(PinholeCameraPtr camera, int max_iter = 10, int max_level = 2, int min_level = 0);
    ~SparseImgAlign();

    Sophus::SE3d Run(const ImagePyr& img_ref_pyr, const ImagePyr& img_cur_pyr,
                     const VecVector2d& ref_pts, const VecVector3d& ref_mps,
                     const Sophus::SE3d& init_Tcr);

private:
    int mMaxLevel; // coarsest pyramid level for the alignment.
    int mMinLevel; // finest pyramid level for the alignment.
    SparseImgAlignImplPtr impl;
};

using SparseImgAlignPtr = std::shared_ptr<SparseImgAlign>;
using SparseImgAlignConstPtr = std::shared_ptr<const SparseImgAlign>;
