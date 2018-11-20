#include "isam2_backend.h"
#include <ros/ros.h>
#include "tracer.h"

ISAM2BackEnd::ISAM2BackEnd(const SimpleStereoCamPtr& camera)
    : mState(INIT), mpCamera(camera)
{
    assert(camera != nullptr);
    mKStereo = boost::make_shared<gtsam::Cal3_S2Stereo>(mpCamera->f, mpCamera->f, 0,
                                                        mpCamera->cx, mpCamera->cy, mpCamera->b);
    mKMono = boost::make_shared<gtsam::Cal3_S2>(mpCamera->f, mpCamera->f, 0, mpCamera->cx, mpCamera->cy);
    gtsam::ISAM2Params parameters;
    parameters.relinearizeThreshold = 0.01;
    parameters.relinearizeSkip = 1;
    mISAM = gtsam::ISAM2(parameters);
}

ISAM2BackEnd::~ISAM2BackEnd() {}

void ISAM2BackEnd::Process() {
    while(1) {
        std::vector<FramePtr> v_keyframe;
        std::unique_lock<std::mutex> lock(mKFBufferMutex);
        mKFBufferCV.wait(lock, [&] {
            v_keyframe = std::vector<FramePtr>(mKFBuffer.begin(), mKFBuffer.end());
            mKFBuffer.clear();
            return !v_keyframe.empty();
        });
        lock.unlock();

        if(mState == INIT) {
            mState = NON_LINEAR;
        }
        else if(mState == NON_LINEAR) {

        }
        else {
            ROS_ERROR_STREAM("BackEnd error state!");
            exit(-1);
        }
    }
}

void ISAM2BackEnd::AddKeyFrame(FramePtr keyframe) {
    assert(keyframe->mIsKeyFrame);
    mKFBufferMutex.lock();
    mKFBuffer.push_back(keyframe);
    mKFBufferMutex.unlock();
    mKFBufferCV.notify_one();
}

void ISAM2BackEnd::CreateMapPointFromStereoMatching(FramePtr keyframe) {
    auto& v_pt_id = keyframe->mvPtID;
    auto& v_uv = keyframe->mv_uv;
    auto& v_ur = keyframe->mv_ur;
    int N = v_uv.size();

    for(int i = 0; i < N; ++i) {
        if(v_ur[i] == -1)
            continue;
        auto it = msIsTriangulate.find(v_pt_id[i]);
        if(it == msIsTriangulate.end())
            continue;

        double disparity = v_uv[i].x - v_ur[i];
        double z = mpCamera->bf / disparity;
        double x = z * (v_uv[i].x - mpCamera->cx) * mpCamera->inv_f;
        double y = z * (v_uv[i].y - mpCamera->cy) * mpCamera->inv_f;

        Eigen::Vector3d x3Dc(x, y, z);
        Eigen::Vector3d x3Dw = keyframe->mTwc * x3Dc;
    }
}
