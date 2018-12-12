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
                if(InitSystem(keyframe)) {
                    mState = NON_LINEAR;
                }
            }
            else if(mState == NON_LINEAR) {
                // create new map point
                CreateMapPointFromStereoMatching(keyframe);
                CreateMapPointFromMotionTracking(keyframe);

                // solve the sliding window BA
                SlidingWindowBA(keyframe);

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
    assert(keyframe->mIsKeyFrame);
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

    ++MapPoint::gTraversalId;
    for(auto& keyframe : v_keyframes) {
        v_Twc.emplace_back(keyframe->mTwc);
        for(auto& mp : keyframe->mvMapPoint) {
            if(!mp || mp->empty() || mp->mTraversalId == MapPoint::gTraversalId)
                continue;
            v_x3Dw.emplace_back(mp->x3Dw());
            mp->mTraversalId = MapPoint::gTraversalId;
        }
    }

    mDebugCallback(v_Twc, v_x3Dw);
}

bool SimpleBackEnd::InitSystem(const FramePtr& keyframe) {
    if(keyframe->mNumStereo < 100)
        return false;
    keyframe->mTwc = Sophus::SE3d();
    CreateMapPointFromStereoMatching(keyframe);
    mpSlidingWindow->push_kf(keyframe);
    return true;
}

void SimpleBackEnd::CreateMapPointFromStereoMatching(const FramePtr& keyframe) {
    ScopedTrace st("c_mappt_s");
    const auto& v_uv = keyframe->mv_uv;
    const auto& v_ur = keyframe->mv_ur;
    const auto& v_mp = keyframe->mvMapPoint;
    const auto& Twc  = keyframe->mTwc;
    const int N = v_uv.size();

    for(int i = 0; i < N; ++i) {
        if(v_ur[i] == -1 || !v_mp[i] || !v_mp[i]->empty())
            continue;

        Eigen::Vector3d p(v_uv[i].x, v_uv[i].y, v_ur[i]);
        Eigen::Vector3d x3Dc;
        mpCamera->Triangulate(p, x3Dc);
        v_mp[i]->Set_x3Dw(x3Dc, Twc);
        mpSlidingWindow->push_mp(v_mp[i]);
    }
}

void SimpleBackEnd::CreateMapPointFromMotionTracking(const FramePtr& keyframe) {
    ScopedTrace st("c_mappt_m");
    const auto& v_mp = keyframe->mvMapPoint;
    const int N = v_mp.size();
    Sophus::SE3d Tw0 = keyframe->mTwc;

    for(auto& mp : v_mp) {
        if(!mp->empty())
            continue;

        auto v_meas = mp->GetMeas();
        int num_meas = v_meas.size();
        if(num_meas < 2)
            continue;

        Eigen::MatrixXd A(2 * num_meas, 4);
        int A_idx = 0;

        for(int i = 0; i < num_meas; ++i) {
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
        }

        // solve AX = 0
        Eigen::Vector4d X = Eigen::JacobiSVD<Eigen::MatrixXd>(A,
                            Eigen::ComputeThinV).matrixV().rightCols<1>();

        X /= X(3);

        if(X(2) < 0.3) // smaller than 30 cm
            continue;

        mp->Set_x3Dw(X.head(3), Tw0);
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

        if(kf == sliding_window[0]) // TODO will remove???
            problem.SetParameterBlockConstant(kf->vertex_data);
    }

    for(auto& mp : mps_in_sliding_window) {
        if(mp->empty())
            continue;
        std::memcpy(mp->vertex_data, mp->x3Dw().data(), sizeof(double) * 3);
        problem.AddParameterBlock(mp->vertex_data, 3); // (optional)

        auto observations = mp->GetMeas();
        for(auto& obs : observations) {
            FramePtr kf = obs.first;
            if(kf->mKeyFrameID > new_keyframe->mKeyFrameID) {
                // this will occur when frontend is faster than backend
                continue;
            }
            size_t idx = obs.second;
            double ur = kf->mv_ur[idx];
            if(ur < 0) { // mono
                if(observations.size() == 1)
                    continue;
                Eigen::Vector2d pt(kf->mv_uv[idx].x, kf->mv_uv[idx].y);
                auto factor = new ProjectionFactor(mpCamera, pt);
                problem.AddResidualBlock(factor, loss_function2, kf->vertex_data, mp->vertex_data);
            }
            else { // stereo
                Eigen::Vector3d pt(kf->mv_uv[idx].x, kf->mv_uv[idx].y, ur);
                auto factor = new StereoProjectionFactor(mpCamera, pt);
                problem.AddResidualBlock(factor, loss_function3, kf->vertex_data, mp->vertex_data);
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
//    std::cout << summary.FullReport() << std::endl;

    for(int i = 0, n = sliding_window.size(); i < n; ++i) {
        Eigen::Map<Sophus::SE3d> Twc(sliding_window[i]->vertex_data);
        sliding_window[i]->mTwc = Twc;
    }

    for(int i = 0, n = mps_in_sliding_window.size(); i < n; ++i) {
        if(mps_in_sliding_window[i]->empty())
            continue;
        Eigen::Map<Eigen::Vector3d> x3Dw(mps_in_sliding_window[i]->vertex_data);
        mps_in_sliding_window[i]->Set_x3Dw(x3Dw);
    }
}
