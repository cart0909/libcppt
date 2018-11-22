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
