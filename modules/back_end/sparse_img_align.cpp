#include "sparse_img_align.h"
#include <ros/ros.h>

SparseImgAlign::SparseImgAlign(PinholeCameraPtr camera, int max_iter, int max_level, int min_level)
    : mMaxLevel(max_level), mMinLevel(min_level)
{
    impl = SparseImgAlignImplPtr(new SparseImgAlignImpl(camera, max_iter));
}

SparseImgAlign::~SparseImgAlign() {

}

Sophus::SE3d SparseImgAlign::Run(const ImagePyr& img_ref_pyr, const ImagePyr& img_cur_pyr,
                                 const VecVector2d& ref_pts, const VecVector3d& ref_mps,
                                 const Sophus::SE3d& init_Tcr)
{
    Sophus::SE3d Tcr = init_Tcr;
    for(int i = mMaxLevel; mMaxLevel >= mMinLevel; --i) {
        auto& img_ref = img_ref_pyr[i];
        auto& img_cur = img_cur_pyr[i];
        impl->Run(i, img_ref, img_cur, ref_pts, ref_mps, Tcr);
    }
    return Tcr;
}
