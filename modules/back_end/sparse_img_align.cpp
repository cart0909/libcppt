#include "sparse_img_align.h"

SparseImgAlign::SparseImgAlign() {

}

SparseImgAlign::~SparseImgAlign() {

}

void SparseImgAlign::Run() {
    int num_pts = mvRefPts.size();
    mRefPatch.create(num_pts, mPatchArea, CV_32F);
    mJpatch_pt.resize(Eigen::NoChange, num_pts * mPatchArea);
    mvIsVisible.resize(num_pts, false);

    for(mCurLevel = mMaxLevel; mCurLevel >= mMinLevel; --mCurLevel) {
        Optimize();
    }
}

bool SparseImgAlign::Solve() {
    mdelx = mH.ldlt().solve(mb);
    if(std::isnan((double)mdelx(0)))
        return false;
    return true;
}

void SparseImgAlign::Update() {
    mTcr = mTcr * Sophus::SE3d::exp(-mdelx);
}

void SparseImgAlign::Optimize() {

}

void SparseImgAlign::ComputeReferencePatchAndJacobian() {
    const int border = mPatchHalfSize + 1;
    const cv::Mat& ref_img = mvRefImgPyr[mCurLevel];
    const int stride = ref_img.cols;
    const float scale = 1.0f / (1 << mCurLevel);
    const Eigen::Vector3d ref_pos = mTrw.inverse().translation();
    const double& fx = mpCamera->fx;
    size_t feature_counter = 0;

    Eigen::Matrix3d Rrw = mTrw.rotationMatrix();
    Eigen::Vector3d trw = mTrw.translation();
    for(int i = 0, n = mvIsVisible.size(); i < n; ++i) {
        const float u_ref = mvRefPts[i].x() * scale;
        const float v_ref = mvRefPts[i].y() * scale;
        const float u_ref_i = std::floor(u_ref);
        const float v_ref_i = std::floor(v_ref);
        if(u_ref_i - border < 0 || v_ref_i - border < 0 ||
           u_ref_i + border >= ref_img.cols || v_ref_i + border >= ref_img.rows)
            continue;
        mvIsVisible[i] = true;

        Eigen::Vector3d x3Dc = Rrw * mvRefMps[i] + trw;
        Eigen::Matrix<double, 2, 3> Jproj;
//        mpCamera->Project()
    }
}
