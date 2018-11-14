#include "sparse_img_align.h"

void SparseImgAlign::SetState(const ImagePyr& ref_img_pyr, const ImagePyr& cur_img_pyr,
                              const VecVector2d& ref_pts, const VecVector3d& ref_mps,
                              const Sophus::SE3d& predict_Tcr) {
    mvRefImgPyr = ref_img_pyr;
    mvCurImgPyr = cur_img_pyr;
    mvRefPts = ref_pts;
    mvRefMps = ref_mps;
    mTcr = predict_Tcr;
}

bool SparseImgAlign::Run(Sophus::SE3d& Tcr) {
    double *Tcr_ptr = mTcr.data();
    for(mCurLevel = mMaxLevel; mCurLevel >= mMinLevel; --mCurLevel) {
        cv::Mat ref_img = mvRefImgPyr[mCurLevel];
        cv::Mat cur_img = mvCurImgPyr[mCurLevel];
        ceres::Problem problem;
        ceres::LocalParameterization* vertex_se3 = Sophus::VertexSE3::Create(false, true);
        problem.AddParameterBlock(Tcr_ptr, Sophus::SE3d::num_parameters, vertex_se3);

        for(int i = 0, n = mvRefPts.size(); i < n; ++i) {
            const auto& x3Dw = mvRefMps[i];
            const auto& uv_r = mvRefPts[i];
            Eigen::Vector3d x3Dr = mTrw * x3Dw;

            auto intensity_factor = new IntensityFactor(mpCamera, ref_img, cur_img, x3Dr,
                                                        uv_r, mCurLevel);
            problem.AddResidualBlock(intensity_factor, NULL, Tcr_ptr);
        }

        ceres::Solver::Options options;
        options.max_num_iterations = mMaxIter;
        options.linear_solver_type = ceres::DENSE_QR;
        options.trust_region_strategy_type = ceres::DOGLEG;

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        summary.BriefReport();
    }
    Tcr = mTcr;
    return true;
}
