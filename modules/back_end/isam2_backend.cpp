#include "isam2_backend.h"
#include <ros/ros.h>
#include <gtsam/inference/Symbol.h>
#include "tracer.h"
using namespace gtsam;

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

        for(auto& keyframe : v_keyframe) {
            if(mState == INIT) {
                if(InitSystem(keyframe)) {
                    mISAM.update(mGraph, mInitValues);
                    Values currentEstimate = mISAM.calculateEstimate();
                    currentEstimate.print("Current estimate: ");
                    mState = NON_LINEAR;
                }
            }
            else if(mState == NON_LINEAR) {

            }
            else {
                ROS_ERROR_STREAM("BackEnd error state!");
                exit(-1);
            }
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
        if(it != msIsTriangulate.end())
            continue;

        msIsTriangulate.insert(v_pt_id[i]);

        double disparity = v_uv[i].x - v_ur[i];
        double z = mpCamera->bf / disparity;
        double x = z * (v_uv[i].x - mpCamera->cx) * mpCamera->inv_f;
        double y = z * (v_uv[i].y - mpCamera->cy) * mpCamera->inv_f;

        Eigen::Vector3d x3Dc(x, y, z);
        Eigen::Vector3d x3Dw = keyframe->mTwc * x3Dc;
        mInitValues.insert(Symbol('l', v_pt_id[i]), Point3(x3Dw(0), x3Dw(1), x3Dw(2)));
    }
}

bool ISAM2BackEnd::InitSystem(FramePtr keyframe) {
    if(keyframe->mNumStereo < 100)
        return false;

    CreateMapPointFromStereoMatching(keyframe);
    // add edge to graph
    mGraph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', keyframe->mKeyFrameID), Pose3());
    mInitValues.insert(Symbol('x', keyframe->mKeyFrameID), Pose3());
    //create factor noise model with 3 sigmas of value 1
    const auto model = noiseModel::Isotropic::Sigma(3,1);

    int N = keyframe->mv_uv.size();
    for(int i = 0; i < N; ++i) {
        if(keyframe->mv_ur[i] == -1)
            continue;
        uint64_t keyframe_id = keyframe->mKeyFrameID;
        uint64_t landmark_id = keyframe->mvPtID[i];
        double u = keyframe->mv_uv[i].x;
        double v = keyframe->mv_uv[i].y;
        double ur = keyframe->mv_ur[i];
        mGraph.emplace_shared<GenericStereoFactor<Pose3, Point3>>(StereoPoint2(u, ur, v), model,
                                                                  Symbol('x', keyframe_id),
                                                                  Symbol('l', landmark_id),
                                                                  mKStereo);
    }
    return true;
}
