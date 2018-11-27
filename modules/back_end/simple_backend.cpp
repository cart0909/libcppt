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
                mpSlidingWindow->push_back(keyframe);
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
    auto v_keyframes = mpSlidingWindow->get();
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
    mpSlidingWindow->push_back(keyframe);
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
        mpSlidingWindow->push_back(v_mp[i]);
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
            auto& uv_i = v_meas[i].second;
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
        mpSlidingWindow->push_back(mp);
    }
}

void SimpleBackEnd::SlidingWindowBA(const FramePtr& new_keyframe) {
    ScopedTrace st("ba");
    std::vector<FramePtr> sliding_window = mpSlidingWindow->get(); // pose vertex
    sliding_window.emplace_back(new_keyframe);
    std::vector<double> keyframes_Tcw_raw((sliding_window.size()) * 7, 0);
    std::vector<MapPointPtr> mps_in_sliding_window; // point vertex
    std::vector<double> mps_x3Dw_raw;

    // edges
    std::vector<std::vector<std::pair<size_t, size_t>>> v_edges; // vector<uv_idx, mappointidx>

    ceres::Problem problem;
    ceres::LossFunction *loss_function2 = new ceres::HuberLoss(5.991);
    ceres::LossFunction *loss_function3 = new ceres::HuberLoss(7.815);
    ceres::LocalParameterization *pose_vertex = new Sophus::VertexSE3(true);

    // traversal
    ++MapPoint::gTraversalId;
    int mps_idx = 0;
    {
        for(int kfs_idx = 0; kfs_idx < sliding_window.size(); ++kfs_idx) {
            std::vector<std::pair<size_t, size_t>> edge;
            Sophus::SE3d Tcw = sliding_window[kfs_idx]->mTwc.inverse();
            std::memcpy(keyframes_Tcw_raw.data() + (7 * kfs_idx), Tcw.data(), sizeof(double) * 7);
            problem.AddParameterBlock(keyframes_Tcw_raw.data() + (7 * kfs_idx), 7, pose_vertex);

            if(kfs_idx == 0) {
                problem.SetParameterBlockConstant(keyframes_Tcw_raw.data());
            }

            auto& v_mappoint_in_kf = sliding_window[kfs_idx]->mvMapPoint;

            for(int i = 0, n = v_mappoint_in_kf.size(); i < n; ++i) {
                auto& mp = v_mappoint_in_kf[i];
                if(!mp || mp->empty())
                    continue;

                if(mp->mTraversalId != MapPoint::gTraversalId) {
                    mp->mTraversalId = MapPoint::gTraversalId;
                    mps_in_sliding_window.emplace_back(mp);
                    mp->mVectorIdx = mps_idx++;
                }

                edge.emplace_back(i, mp->mVectorIdx);
            }

            v_edges.emplace_back(edge);
        }

        mps_x3Dw_raw.resize(mps_idx * 3, 0);
        for(int i = 0, n = mps_in_sliding_window.size(); i < n; ++i) {
            auto& mp = mps_in_sliding_window[i];
            auto x3Dw = mp->x3Dw();
            std::memcpy(mps_x3Dw_raw.data() + (i * 3), x3Dw.data(), sizeof(double) * 3);
            problem.AddParameterBlock(mps_x3Dw_raw.data() + (i * 3), 3);
//            problem.SetParameterBlockConstant(mps_x3Dw_raw.data() + (i * 3));
        }
    }

//    for(auto& edge : v_edges) {
//        std::cout << edge.size() << " ";
//    }
//    std::cout << "total: " << mps_idx << std::endl;

    for(int i = 0, n = sliding_window.size(); i < n; ++i) {
        auto& edges = v_edges[i];
        double* pose_raw = keyframes_Tcw_raw.data() + 7 * i;

        for(auto& it : edges) {
            auto& uv = sliding_window[i]->mv_uv[it.first];
            double ur = sliding_window[i]->mv_ur[it.first];
            auto& mp = mps_in_sliding_window[it.second];
            double* point_raw = mps_x3Dw_raw.data() + 3 * it.second;
            if(ur < 0) { // mono constrain
                ProjectionFactor* project_factor =
                        new ProjectionFactor(mpCamera, Eigen::Vector2d(uv.x, uv.y));
                problem.AddResidualBlock(project_factor, loss_function2, pose_raw, point_raw);
            }
            else { // stereo constrain
                StereoProjectionFactor* project_factor =
                        new StereoProjectionFactor(mpCamera, Eigen::Vector3d(uv.x, uv.y, ur));
                problem.AddResidualBlock(project_factor, loss_function3, pose_raw, point_raw);
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
        Eigen::Map<Sophus::SE3d> Tcw(keyframes_Tcw_raw.data() + (7 * i));
        sliding_window[i]->mTwc = Tcw.inverse();
    }

    for(int i = 0, n = mps_in_sliding_window.size(); i < n; ++i) {
        Eigen::Map<Eigen::Vector3d> x3Dw(mps_x3Dw_raw.data() + (3 * i));
        mps_in_sliding_window[i]->Set_x3Dw(x3Dw);
    }
}
