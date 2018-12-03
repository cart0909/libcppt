#include "simple_frontend.h"
#include "basic_datatype/tic_toc.h"
#include "basic_datatype/util_datatype.h"
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include "tracer.h"
#include <ros/ros.h>

SimpleFrontEnd::SimpleFrontEnd(const SimpleStereoCamPtr& camera,
                               const SlidingWindowPtr& sliding_window)
    : mpCamera(camera), mpSldingWindow(sliding_window)
{

}

SimpleFrontEnd::~SimpleFrontEnd() {

}

// with frame without any feature extract by FAST corner detector
// if exist some features extract FAST corner in empty grid.
void SimpleFrontEnd::ExtractFeatures(const FramePtr& frame) {
    ScopedTrace st("EFeat");
    // uniform the feature distribution
//    UniformFeatureDistribution(frame); // FIXME!!!

    static int empty_value = -1, exist_value = -2;
    int grid_rows = std::ceil(static_cast<float>(mpCamera->height) / 32);
    int grid_cols = std::ceil(static_cast<float>(mpCamera->width) / 32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = frame->mv_uv.size(); i < n; ++i) {
        auto& pt = frame->mv_uv[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grids[grid_i][grid_j] != exist_value)
            grids[grid_i][grid_j] = exist_value;
    }

    std::vector<cv::KeyPoint> kps;
    Tracer::TraceBegin("FAST");
    cv::FAST(frame->mImgL, kps, 20);
    Tracer::TraceEnd();

    for(int i = 0, n = kps.size(); i < n; ++i) {
        auto& pt = kps[i].pt;
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        if(grid_i >= grid_rows)
            grid_i = grid_rows - 1;
        if(grid_j >= grid_cols)
            grid_j = grid_cols - 1;
        int value = grids[grid_i][grid_j];
        if(value != exist_value) {
            if(value == empty_value) {
                grids[grid_i][grid_j] = i;
            }
            else {
                if(kps[i].response > kps[value].response)
                    grids[grid_i][grid_j] = i;
            }
        }
    }

    for(auto& i : grids)
        for(auto& j : i)
            if(j >= 0) {
                frame->mvPtCount.emplace_back(0);
                frame->mv_uv.emplace_back(kps[j].pt);
                frame->mvLastKFuv.emplace_back(kps[j].pt);
                frame->mvMapPoint.emplace_back(new MapPoint);
            }
}

// track features by optical flow and check epipolar constrain
void SimpleFrontEnd::TrackFeaturesByOpticalFlow(const FrameConstPtr& ref_frame,
                                                const FramePtr& cur_frame) {
    if(ref_frame->mv_uv.empty())
        return;
    ScopedTrace st("TrackFeat");

    const int width = mpCamera->width, height = mpCamera->height;

    // copy ref frame data for cur frame
    std::vector<cv::Point2f> ref_frame_pts = ref_frame->mv_uv;
    cur_frame->mvPtCount = ref_frame->mvPtCount;
    cur_frame->mvMapPoint = ref_frame->mvMapPoint;
    if(ref_frame->mIsKeyFrame) {
        cur_frame->mvLastKFuv = ref_frame->mv_uv;
    }
    else {
        cur_frame->mvLastKFuv = ref_frame->mvLastKFuv;
    }

    // optical flow
    std::vector<uchar> status;
    std::vector<float> err;
    Tracer::TraceBegin("Optical Flow");
    cv::calcOpticalFlowPyrLK(ref_frame->mImgPyrGradL, cur_frame->mImgPyrGradL, ref_frame_pts,
                             cur_frame->mv_uv, status, err, cv::Size(21, 21), 3);
    Tracer::TraceEnd();

    for(int i = 0, n = cur_frame->mv_uv.size(); i < n; ++i)
        if(status[i] && !InBorder(cur_frame->mv_uv[i], width, height))
            status[i] = 0;

    Tracer::TraceBegin("reduce vector");
    ReduceVector(ref_frame_pts, status);
    ReduceVector(cur_frame->mv_uv, status);
    ReduceVector(cur_frame->mvPtCount, status);
    ReduceVector(cur_frame->mvLastKFuv, status);
    ReduceVector(cur_frame->mvMapPoint, status);
    Tracer::TraceEnd();

    RemoveOutlierFromF(ref_frame_pts, cur_frame);

    for(auto& it : cur_frame->mvPtCount)
        ++it;
}

void SimpleFrontEnd::RemoveOutlierFromF(std::vector<cv::Point2f>& ref_pts,
                                        const FramePtr& cur_frame) {
    if(ref_pts.size() > 8) {
        ScopedTrace st("RemoveF");
        std::vector<uchar> status;
        cv::findFundamentalMat(ref_pts, cur_frame->mv_uv, cv::FM_RANSAC, 1, 0.99, status);
        ReduceVector(ref_pts, status);
        ReduceVector(cur_frame->mv_uv, status);
        ReduceVector(cur_frame->mvPtCount, status);
        ReduceVector(cur_frame->mvLastKFuv, status);
        ReduceVector(cur_frame->mvMapPoint, status);
    }
}

void SimpleFrontEnd::UniformFeatureDistribution(const FramePtr& cur_frame) {
    ScopedTrace st("UDist");
    static int empty_value = -1;
    int grid_rows = std::ceil(static_cast<float>(mpCamera->height)/32);
    int grid_cols = std::ceil(static_cast<float>(mpCamera->width)/32);
    std::vector<std::vector<int>> grids(grid_rows, std::vector<int>(grid_cols, empty_value));
    for(int i = 0, n = cur_frame->mv_uv.size(); i < n; ++i) {
        auto& pt = cur_frame->mv_uv[i];
        int grid_i = pt.y / 32;
        int grid_j = pt.x / 32;
        int idx = grids[grid_i][grid_j];
        if(idx != empty_value) {
            if(cur_frame->mvPtCount[i] > cur_frame->mvPtCount[idx])
                grids[grid_i][grid_j] = i;
        }
        else
            grids[grid_i][grid_j] = i;
    }

    std::vector<uint64_t> temp_pt_id;
    std::vector<uint32_t> temp_pt_count;
    std::vector<cv::Point2f> temp_pts;
    std::vector<cv::Point2f> temp_pts_lastkf;
    std::vector<MapPointPtr> temp_mps;
    for(auto& i : grids) {
        for(auto& j : i) {
            if(j != empty_value) {
                temp_pt_count.emplace_back(cur_frame->mvPtCount[j]);
                temp_pts.emplace_back(cur_frame->mv_uv[j]);
                temp_pts_lastkf.emplace_back(cur_frame->mvLastKFuv[j]);
                temp_mps.emplace_back(cur_frame->mvMapPoint[j]);
            }
        }
    }

    cur_frame->mvPtCount = std::move(temp_pt_count);
    cur_frame->mv_uv = std::move(temp_pts);
    cur_frame->mvLastKFuv = std::move(temp_pts_lastkf);
    cur_frame->mvMapPoint = std::move(temp_mps);
}

void SimpleFrontEnd::PoseOpt(const FramePtr& cur_frame, const Sophus::SE3d& init_Tcw) {
    ScopedTrace st("poseopt");
    auto& v_uv = cur_frame->mv_uv;
    auto& v_ur = cur_frame->mv_ur;
    auto& v_mps = cur_frame->mvMapPoint;

    int N = v_uv.size();
    std::vector<uchar> status(N, 1);
    std::vector<ceres::CostFunction*> cost_fcuntion(N, nullptr);

    double Tcw_raw[7];
    std::memcpy(Tcw_raw, init_Tcw.data(), sizeof(double) * 7);

    ceres::Problem problem;
    const double mono_chi2 = 5.991, stereo_chi2 = 7.815;
    ceres::LossFunction* loss_function2 = new ceres::HuberLoss(std::sqrt(mono_chi2));
    ceres::LossFunction* loss_function3 = new ceres::HuberLoss(std::sqrt(stereo_chi2));
    ceres::LocalParameterization* pose_vertex = new Sophus::VertexSE3(true);

    problem.AddParameterBlock(Tcw_raw, 7, pose_vertex);

    for(int i = 0; i < N; ++i) {
        if(!v_mps[i] || v_mps[i]->empty())
            continue;

        if(1) { // mono edge
            auto project_factor =
            new unary::ProjectionFactor(mpCamera,
                                        Eigen::Vector2d(v_uv[i].x, v_uv[i].y),
                                        v_mps[i]->x3Dw());
            problem.AddResidualBlock(project_factor, loss_function2, Tcw_raw);
            cost_fcuntion[i] = project_factor;
        }
        else { // stereo edge
            auto stereo_project_factor =
            new unary::StereoProjectionFactor(mpCamera,
                                              Eigen::Vector3d(v_uv[i].x, v_uv[i].y, v_ur[i]),
                                              v_mps[i]->x3Dw());
            problem.AddResidualBlock(stereo_project_factor, loss_function3, Tcw_raw);
            cost_fcuntion[i] = stereo_project_factor;
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
//    std::cout << summary.FullReport() << std::endl;

    // remove outlier
    for(int i = 0; i < N; ++i) {
        if(cost_fcuntion[i]) {
            if(1) { // mono
                double* parameters[1] = {Tcw_raw};
                Eigen::Vector2d residual;
                cost_fcuntion[i]->Evaluate(parameters, residual.data(), nullptr);
                if(residual.norm() > mono_chi2) {
                    status[i] = 0;
                }
            }
            else { // stereo
                double* parameters[1] = {Tcw_raw};
                Eigen::Vector3d residual;
                cost_fcuntion[i]->Evaluate(parameters, residual.data(), nullptr);
                if(residual.norm() > stereo_chi2) {
                    status[i] = 0;
                }
            }
        }
    }
    // set pose to cur frame
    Eigen::Map<Sophus::SE3d> opt_Tcw(Tcw_raw);
    cur_frame->mTwc = opt_Tcw.inverse();

    ReduceVector(cur_frame->mv_uv, status);
    ReduceVector(cur_frame->mv_ur, status);
    ReduceVector(cur_frame->mvPtCount, status);
    ReduceVector(cur_frame->mvLastKFuv, status);
    ReduceVector(cur_frame->mvMapPoint, status);
}
