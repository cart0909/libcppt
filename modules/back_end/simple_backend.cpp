#include "simple_backend.h"
#include <ceres/ceres.h>
#include "ceres/local_parameterization_se3.h"
#include "ceres/projection_factor.h"
#include <ros/ros.h>
#include "tracer.h"

SimpleBackEnd::SimpleBackEnd(const SimpleStereoCamPtr& camera,
                             const SlidingWindowPtr& sliding_window)
    : mState(INIT), mpCamera(camera), mpSlidingWindow(sliding_window)
{}

SimpleBackEnd::~SimpleBackEnd() {}

void SimpleBackEnd::Process() {
    while(1) {
        std::vector<FramePtr> v_keyframe;
        std::unique_lock<std::mutex> lock(mKFBufferMutex);
        mKFBufferCV.wait(lock, [&]{
           v_keyframe = mKFBuffer;
           mKFBuffer.clear();
           return !v_keyframe.empty();
        });
        lock.unlock();

        for(auto& keyframe : v_keyframe) {
            if(mState == INIT) {
                if(InitSystem(keyframe))
                    mState = NON_LINEAR;
                else
                    keyframe->set_bad();
            }
            else if(mState == NON_LINEAR) {
                // create new map point
                CreateMapPoint(keyframe);

                // solve the sliding window BA
                SlidingWindowBA(keyframe);
                Marginalization(keyframe);

                // add to sliding window
                mpSlidingWindow->push_kf(keyframe);
            }
            else {
                ROS_ERROR_STREAM("back end error state!!! exit the program.");
                exit(-1);
            }
        }

        ShowResultGUI();
    }
}

void SimpleBackEnd::AddKeyFrame(const FramePtr& keyframe) {
    mKFBufferMutex.lock();
    mKFBuffer.push_back(keyframe);
    mKFBufferMutex.unlock();
    mKFBufferCV.notify_one();
}

void SimpleBackEnd::SetDebugCallback(
        const std::function<void(const std::vector<Sophus::SE3d>&,
                            const VecVector3d&)>& callback)
{
    mDebugCallback = callback;
}

void SimpleBackEnd::ShowResultGUI() const {
    if(!mDebugCallback)
        return;
    auto v_keyframes = mpSlidingWindow->get_kfs();
    std::vector<Sophus::SE3d> v_Twc;
    VecVector3d v_x3Dw;

    auto v_mps = mpSlidingWindow->get_mps();

    for(auto& keyframe : v_keyframes) {
        v_Twc.emplace_back(keyframe->mTwc);
    }

    for(auto& mp : v_mps) {
        if(!mp->is_init())
            continue;
        v_x3Dw.emplace_back(mp->x3Dw());
    }

    mDebugCallback(v_Twc, v_x3Dw);
}

bool SimpleBackEnd::InitSystem(const FramePtr& keyframe) {
    if(keyframe->mNumStereo < 100)
        return false;
    keyframe->mTwc = Sophus::SE3d();
    CreateMapPoint(keyframe);
    mpSlidingWindow->push_kf(keyframe);
    return true;
}

void SimpleBackEnd::CreateMapPoint(const FramePtr& keyframe) {
    ScopedTrace st("create_mps");
    const auto& v_mp = keyframe->mvMapPoint;

    for(auto& mp : v_mp) {
        if(mp->is_init() || mp->is_bad())
            continue;

        auto v_meas = mp->get_meas(keyframe);

        if(v_meas.empty())
            continue;

        int num_meas = v_meas.size();
        for(auto& it : v_meas) {
            auto& kf  = it.first;
            auto& idx = it.second;
            if(kf->mv_ur[idx] != -1) {
                ++num_meas;
            }
        }

        if(num_meas < 2)
            continue;

        Sophus::SE3d Tw0 = mp->get_parent().first->mTwc;// parent
        Eigen::MatrixXd A(2 * num_meas, 4);
        int A_idx = 0;

        for(int i = 0; i < v_meas.size(); ++i) {
            auto& keyframe_i = v_meas[i].first;
            auto& uv_i = keyframe_i->mv_uv[v_meas[i].second];
            const Sophus::SE3d& Tiw = keyframe_i->mTwc.inverse();
            Sophus::SE3d Ti0 = Tiw * Tw0;
            Eigen::Matrix<double, 3, 4> P;
            P << Ti0.rotationMatrix(), Ti0.translation();
            Eigen::Vector3d normal_plane_uv;
            mpCamera->BackProject(Eigen::Vector2d(uv_i.x, uv_i.y), normal_plane_uv);
            A.row(A_idx++) = P.row(0) - normal_plane_uv(0) * P.row(2);
            A.row(A_idx++) = P.row(1) - normal_plane_uv(1) * P.row(2);

            auto& ur_i = keyframe_i->mv_ur[v_meas[i].second];
            if(ur_i != -1) {
                Sophus::SE3d Trl = Sophus::SE3d::transX(-mpCamera->b);
                Sophus::SE3d Tri_0 = Trl * Ti0;
                P << Tri_0.rotationMatrix(), Tri_0.translation();
                mpCamera->BackProject(Eigen::Vector2d(ur_i, uv_i.y), normal_plane_uv);
                A.row(A_idx++) = P.row(0) - normal_plane_uv(0) * P.row(2);
                A.row(A_idx++) = P.row(1) - normal_plane_uv(1) * P.row(2);
            }
        }

        // solve AX = 0
        Eigen::Vector4d X = Eigen::JacobiSVD<Eigen::MatrixXd>(A,
                            Eigen::ComputeThinV).matrixV().rightCols<1>();

        X /= X(3);

        if(X(2) < 0.3) // smaller than 30 cm
            continue;

        mp->inv_z(1.0 / X(2));
        mpSlidingWindow->push_mp(mp);
    }
}

void SimpleBackEnd::SlidingWindowBA(const FramePtr& new_keyframe) {
    ScopedTrace st("ba");
    std::vector<FramePtr> sliding_window = mpSlidingWindow->get_kfs(); // pose vertex
    sliding_window.emplace_back(new_keyframe);
    std::vector<MapPointPtr> mps_in_sliding_window = mpSlidingWindow->get_mps(); // point vertex

    ceres::Problem problem;
    ceres::LossFunction *loss_function2 = new ceres::CauchyLoss(std::sqrt(5.991));
    ceres::LossFunction *loss_function3 = new ceres::CauchyLoss(std::sqrt(7.815));
    ceres::LocalParameterization *pose_vertex = new Sophus::VertexSE3();

    // set pose vertex
    for(auto& kf : sliding_window) {
        std::memcpy(kf->vertex_data, kf->mTwc.data(), sizeof(double) * 7);
        problem.AddParameterBlock(kf->vertex_data, 7, pose_vertex);

        if(kf->mKeyFrameID == 0)
            problem.SetParameterBlockConstant(kf->vertex_data);
    }

    // prior
    if(mpMarginInfo) {
        auto factor = new MarginalizationFactor(mpMarginInfo);
        problem.AddResidualBlock(factor, NULL, mvMarginParameterBlock);
    }

    for(auto& mp : mps_in_sliding_window) {
        if(!mp->is_init() || mp->is_bad())
            continue;
        mp->vertex_data[0] = mp->inv_z();
//        problem.AddParameterBlock(mp->vertex_data, 1); // (optional)

        auto observations = mp->get_meas(new_keyframe);

        if(observations.empty())
            continue;

        auto parent_kf_idx = mp->get_parent();
        FramePtr& parent_kf = parent_kf_idx.first;
        size_t parent_idx = parent_kf_idx.second;
        Eigen::Vector2d pt_i(parent_kf->mv_uv[parent_idx].x, parent_kf->mv_uv[parent_idx].y);

        for(auto& obs : observations) {
            FramePtr kf = obs.first;
            if(kf->mKeyFrameID > new_keyframe->mKeyFrameID) {
                // this will occur when frontend is faster than backend
                continue;
            }
            size_t idx = obs.second;
            double ur = kf->mv_ur[idx];
            if(ur < 0) { // mono
                if(kf == parent_kf)
                    continue;
                Eigen::Vector2d pt_j(kf->mv_uv[idx].x, kf->mv_uv[idx].y);
                auto factor = new ProjectionFactor(mpCamera, pt_j, pt_i);
                problem.AddResidualBlock(factor, loss_function2, kf->vertex_data, mp->vertex_data,
                                         parent_kf->vertex_data);
            }
            else { // stereo
                if(kf == parent_kf) // FIXME
                    continue;
                Eigen::Vector3d pt_j(kf->mv_uv[idx].x, kf->mv_uv[idx].y, ur);
                auto factor = new StereoProjectionFactor(mpCamera, pt_j, pt_i);
                problem.AddResidualBlock(factor, loss_function3, kf->vertex_data, mp->vertex_data,
                                         parent_kf->vertex_data);
            }
        }
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.trust_region_strategy_type = ceres::DOGLEG;
    options.max_num_iterations = 10;
    options.num_threads = 4;
    ceres::Solver::Summary summary;

    ceres::Solve(options, &problem, &summary);
    std::cout << summary.FullReport() << std::endl;

    // restore the optimization data to pose and point
    for(int i = 0, n = sliding_window.size(); i < n; ++i) {
        Eigen::Map<Sophus::SE3d> Twc(sliding_window[i]->vertex_data);
        sliding_window[i]->mTwc = Twc;
    }

    for(int i = 0, n = mps_in_sliding_window.size(); i < n; ++i) {
        if(!mps_in_sliding_window[i]->is_init() || mps_in_sliding_window[i]->is_bad())
            continue;
        mps_in_sliding_window[i]->inv_z(mps_in_sliding_window[i]->vertex_data[0]);
    }
}

void SimpleBackEnd::Marginalization(const FramePtr& new_keyframe) {
    if(mpSlidingWindow->size_kfs() < mpSlidingWindow->max_len)
        return;

    Tracer::TraceBegin("prepare");
    MarginalizationInfo* margin_info = new MarginalizationInfo();
    FramePtr margin_out_kf = mpSlidingWindow->front_kf();
    std::memcpy(margin_out_kf->vertex_data, margin_out_kf->mTwc.data(), sizeof(double) * 7);

    if(mpMarginInfo) { // last_marginalization_info
        std::vector<int> drop_set;
        for(int i = 0, n = mvMarginParameterBlock.size(); i < n; ++i) {
            if(mvMarginParameterBlock[i] == margin_out_kf->vertex_data) {
                drop_set.emplace_back(i);
            }
        }
        auto factor = new MarginalizationFactor(mpMarginInfo);
        auto residual_block_info = new ResidualBlockInfo(factor, nullptr,
                                                         mvMarginParameterBlock,
                                                         drop_set);
        margin_info->addResidualBlockInfo(residual_block_info);
    }

    const double mono_chi2 = 5.991, stereo_chi2 = 7.815;
    ceres::LossFunction *loss_function2(new ceres::CauchyLoss(std::sqrt(mono_chi2))),
                        *loss_function3(new ceres::CauchyLoss(std::sqrt(stereo_chi2)));

    for(auto &mp : margin_out_kf->mvMapPoint) {
        if(!mp->is_init() || mp->is_bad())
            continue;

        auto parent_kf_idx = mp->get_parent();
        if(parent_kf_idx.first != margin_out_kf) {
            continue;
        }
        mp->vertex_data[0] = mp->inv_z();

        FramePtr& kf_i = parent_kf_idx.first;
        size_t idx_i = parent_kf_idx.second;
        Eigen::Vector2d pt_i(kf_i->mv_uv[idx_i].x, kf_i->mv_uv[idx_i].y);

        auto measures = mp->get_meas(new_keyframe);

        for(auto& meas : measures) {
            FramePtr& kf_j = meas.first;

            if(kf_j->mKeyFrameID > new_keyframe->mKeyFrameID) {
                // this will occur when frontend is faster than backend
                continue;
            }

            size_t idx_j = meas.second;
            if(kf_i == kf_j)
                continue;

            double ur = kf_j->mv_ur[idx_j];

            if(ur < 0) { // mono
                Eigen::Vector2d pt_j(kf_j->mv_uv[idx_j].x, kf_j->mv_uv[idx_j].y);
                auto factor = new ProjectionFactor(mpCamera, pt_j, pt_i);
                auto residual_block_info = new ResidualBlockInfo(factor, loss_function2,
                                                                 std::vector<double*>{kf_j->vertex_data, mp->vertex_data, kf_i->vertex_data},
                                                                 std::vector<int>{1, 2});
                margin_info->addResidualBlockInfo(residual_block_info);
            }
            else { // stereo
                Eigen::Vector3d pt_j(kf_j->mv_uv[idx_j].x, kf_j->mv_uv[idx_j].y, ur);
                auto factor = new StereoProjectionFactor(mpCamera, pt_j, pt_i);
                auto residual_block_info = new ResidualBlockInfo(factor, loss_function3,
                                                                 std::vector<double*>{kf_j->vertex_data, mp->vertex_data, kf_i->vertex_data},
                                                                 std::vector<int>{1, 2});
                margin_info->addResidualBlockInfo(residual_block_info);
            }
        }
    }
    Tracer::TraceEnd();
    margin_info->preMarginalize();
    margin_info->marginalize();
    mvMarginParameterBlock = margin_info->getParameterBlocks();

    if(mpMarginInfo)
        delete mpMarginInfo;
    mpMarginInfo = margin_info;
    mpSlidingWindow->front_kf()->set_bad();
}
