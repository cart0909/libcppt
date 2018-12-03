#include "sparse_img_align.h"
#include <ros/ros.h>
#include "tracer.h"

#define BILINEAR(tl_, tr_, bl_, br_, img_ptr_, stride_, u_shift_, v_shift_) \
    tl_ * img_ptr_[u_shift_ + v_shift_ * stride_] + tr_ * img_ptr_[u_shift_ + v_shift_ * stride_ + 1] + \
    bl_ * img_ptr_[u_shift_ + v_shift_ * stride_ + stride_] + br_ * img_ptr_[u_shift_ + v_shift_ * stride_ + stride_ + 1]

const int SparseImgAlign::mPatchSize = 4, SparseImgAlign::mPatchHalfSize = 2, SparseImgAlign::mPatchArea = 16;

SparseImgAlign::SparseImgAlign(const SimpleStereoCamPtr& camera)
    : mpCamera(camera), mMaxIter(10)
{}
SparseImgAlign::~SparseImgAlign() {}

// return estimate_Tcr
Sophus::SE3d SparseImgAlign::Run(const FramePtr& cur_frame, const FramePtr& ref_frame,
                                 Sophus::SE3d init_Tcr) {
    mvRef_uv.clear();
    mv_x3Dr.clear();
    mImgPyrRef = ref_frame->mImgPyrL;
    mImgPyrCur = cur_frame->mImgPyrL;
    Sophus::SE3d Trw = ref_frame->mTwc.inverse();

    for(int i = 0, n = ref_frame->mv_uv.size(); i < n; ++i) {
        const auto& uv = ref_frame->mv_uv[i];
        const auto& mp = ref_frame->mvMapPoint[i];
        if(mp->empty())
            continue;
        mvRef_uv.emplace_back(uv);
        mv_x3Dr.emplace_back(mp->x3Dc(Trw));
    }
    mvVisible.resize(mv_x3Dr.size(), false);
    mJacobianCache.resize(mv_x3Dr.size() * mPatchArea, Eigen::NoChange);
    mRefPatchCache.resize(mPatchArea, mv_x3Dr.size());

    mTcr = init_Tcr;
    for(mLevel = mImgPyrCur.size() - 1; mLevel >= 0; --mLevel) {
        std::cout << "-------------level " << mLevel << std::endl;
        Solve();
    }
    return mTcr;
}

void SparseImgAlign::Solve() {
    mJacobianCache.setZero();
    mRefPatchCache.setZero();
    mvVisible.resize(mvVisible.size(), false);

    PrecomputeCache();

    for(int iter = 0; iter < mMaxIter; ++iter) {
        std::cout << "iter " << iter << ":" << ComputeResidual() << std::endl;
        SolveHxbUpdate();
    }
}

void SparseImgAlign::PrecomputeCache() {
    ScopedTrace st(("pre" + std::to_string(mLevel)).c_str());
    const int border = mPatchHalfSize + 1;
    const cv::Mat& ref_img = mImgPyrRef[mLevel];
    const int stride = ref_img.cols;
    const float scale = 1.0/(1 << mLevel);
    int num_visible = 0;

    for(int i = 0, n = mvRef_uv.size(); i < n; ++i) {
        const float u_ref = mvRef_uv[i].x * scale;
        const float v_ref = mvRef_uv[i].y * scale;
        const int u_ref_i = std::floor(u_ref);
        const int v_ref_i = std::floor(v_ref);

        if(u_ref_i - border < 0 || v_ref_i - border < 0 || u_ref_i + border >= ref_img.cols ||
                v_ref_i + border >= ref_img.rows)
            continue;
        ++num_visible;
        mvVisible[i] = true;

        // projection jacobian
        Eigen::Matrix<double, 2, 3> Jpi_x3Dr = mpCamera->J2(mv_x3Dr[i]);
        Eigen::Matrix<double, 3, 6> Jx3Dr_pose;
        Jx3Dr_pose << Eigen::Matrix3d::Identity(), -Sophus::SO3d::hat(mv_x3Dr[i]);
        Eigen::Matrix<double, 2, 6> Jpi_pose = Jpi_x3Dr * Jx3Dr_pose;

        const float subpix_u_ref = u_ref - u_ref_i;
        const float subpix_v_ref = v_ref - v_ref_i;
        const float w_ref_tl = (1.0 - subpix_u_ref) * (1.0 - subpix_v_ref);
        const float w_ref_tr = subpix_u_ref * (1.0 - subpix_v_ref);
        const float w_ref_bl = (1.0 - subpix_u_ref) * subpix_v_ref;
        const float w_ref_br = subpix_u_ref * subpix_v_ref;

        float* ref_patch = mRefPatchCache.data() + i * mPatchArea;
        for(int y = 0; y < mPatchSize; ++y) {
            int vbegin = v_ref_i - mPatchHalfSize;
            int ubegin = u_ref_i - mPatchHalfSize;
            uchar* ref_img_ptr = ref_img.data + (vbegin + y) * stride + ubegin;
            for(int x = 0; x < mPatchSize; ++x, ++ref_img_ptr, ++ref_patch) {
                *ref_patch = BILINEAR(w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br,
                                      ref_img_ptr, stride, 0, 0);

                float dx = 0.5 * ((BILINEAR(w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br,
                                           ref_img_ptr, stride, 1, 0)) -
                                  (BILINEAR(w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br,
                                           ref_img_ptr, stride, -1, 0)));
                float dy = 0.5 * ((BILINEAR(w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br,
                                            ref_img_ptr, stride, 0, 1)) -
                                  (BILINEAR(w_ref_tl, w_ref_tr, w_ref_bl, w_ref_br,
                                            ref_img_ptr, stride, 0, -1)));

                mJacobianCache.row(i * mPatchArea + y * mPatchSize + x)
                        = (dx * Jpi_pose.row(0) + dy * Jpi_pose.row(1)) * scale;
            }
        }
    }

    std::cout << "num_visible " << num_visible << std::endl;
}

double SparseImgAlign::ComputeResidual() {
    mH.setZero();
    mb.setZero();
    const cv::Mat& cur_img = mImgPyrCur[mLevel];
    const int stride = cur_img.cols;
    const int border = mPatchHalfSize + 1;
    const float scale = 1.0 / (1 << mLevel);
    double chi2 = 0.0;
    int num_pixel = 0;

    for(int i = 0, n = mvRef_uv.size(); i < n; ++i) {
        if(!mvVisible[i])
            continue;
        Eigen::Vector3d x3Dc = mTcr * mv_x3Dr[i];
        Eigen::Vector2d uv;
        mpCamera->Project2(x3Dc, uv);
        uv *= scale;
        const float u_cur = uv(0);
        const float v_cur = uv(1);
        const int u_cur_i = std::floor(u_cur);
        const int v_cur_i = std::floor(v_cur);

        if(u_cur_i - border < 0 || v_cur_i - border < 0 || u_cur_i + border >= cur_img.cols ||
                v_cur_i + border >= cur_img.rows)
            continue;

        const float subpix_u_cur = u_cur - u_cur_i;
        const float subpix_v_cur = v_cur - v_cur_i;
        const float w_cur_tl = (1.0 - subpix_u_cur) * (1.0 - subpix_v_cur);
        const float w_cur_tr = subpix_u_cur * (1.0 - subpix_v_cur);
        const float w_cur_bl = (1.0 - subpix_u_cur) * subpix_v_cur;
        const float w_cur_br = subpix_u_cur * subpix_v_cur;
        float* ref_patch = mRefPatchCache.data() + i * mPatchArea;

//        {
//            cv::Mat debug_ref_patch(4, 4, CV_32F, ref_patch);
////            std::cout << mRefPatchCache << std::endl;
//            debug_ref_patch.convertTo(debug_ref_patch, CV_8U);
//            cv::namedWindow("ref_patch", cv::WINDOW_FREERATIO);
//            cv::imshow("ref_patch", debug_ref_patch);
//            cv::waitKey(0);
//        }

        for(int y = 0; y < mPatchSize; ++y) {
            int ubegin = u_cur_i - mPatchHalfSize;
            int vbegin = v_cur_i - mPatchHalfSize;
            uchar* cur_img_ptr = cur_img.data + (vbegin + y) * stride + ubegin;
            for(int x = 0; x < mPatchSize; ++x, ++cur_img_ptr, ++ref_patch) {
                const float intensity = BILINEAR(w_cur_tl, w_cur_tr, w_cur_bl, w_cur_br,
                                                 cur_img_ptr, stride, 0, 0);
                const float res = intensity - *ref_patch;

                chi2 += res * res;
                ++num_pixel;
                Eigen::Matrix<double, 1, 6> J(mJacobianCache.row(i * mPatchArea + y * mPatchSize + x));
                mH += J.transpose()*J;
                mb -= J.transpose()*res;
            }
        }
    }

    return chi2/num_pixel;
}

void SparseImgAlign::SolveHxbUpdate() {
    // solve linear equation by QR decomposition
    Eigen::Matrix<double, 6, 1> x = mH.colPivHouseholderQr().solve(mb);
    if(std::isnan(x(0)))
        return;
    // update by inverse compositional
    mTcr = mTcr * Sophus::SE3d::exp(x);
}

#undef BILINEAR
