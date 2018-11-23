#include "simple_backend.h"
#include <ros/ros.h>

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
                // solve the linear system get init pose
                SolvePnP(keyframe);
                // create new map point
                CreateMapPointFromStereoMatching(keyframe);
                CreateMapPointFromMotionTracking(keyframe);
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
    }
}

void SimpleBackEnd::CreateMapPointFromMotionTracking(const FramePtr& keyframe) {
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
    }
}

bool SimpleBackEnd::SolvePnP(const FramePtr& keyframe) {
    auto& v_pt = keyframe->mv_uv;
    auto& v_mp = keyframe->mvMapPoint;
    int N = keyframe->mv_uv.size();
    std::vector<cv::Point2f> image_points;
    std::vector<cv::Point3f> object_points;
    double f = mpCamera->f;
    double cx = mpCamera->cx;
    double cy = mpCamera->cy;
    cv::Mat K = (cv::Mat_<double>(3, 3) << f, 0, cx,
                                           0, f, cy,
                                           0, 0,  0);

    for(int i = 0; i < N; ++i) {
        if(!v_mp[i] || v_mp[i]->empty())
            continue;
        Eigen::Vector3d x3Dw = v_mp[i]->x3Dw();
        object_points.emplace_back(x3Dw.x(), x3Dw.y(), x3Dw.z());
        image_points.emplace_back(v_pt[i].x, v_pt[i].y);
    }

    if(image_points.size() > 16) {
        cv::Mat rvec, tvec, R;
        cv::solvePnPRansac(object_points, image_points, K, cv::noArray(), rvec, tvec, false,
                           100, 8.0, 0.99, cv::noArray(), cv::SOLVEPNP_EPNP);
        cv::Rodrigues(rvec, R);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eigen_R(R.ptr<double>());
        Eigen::Map<Eigen::Vector3d> eigen_t(tvec.ptr<double>());
        Sophus::SE3d Tcw(eigen_R, eigen_t), Twc = Tcw.inverse();
        keyframe->mTwc = Twc;
        return true;
    }
    return false;
}
