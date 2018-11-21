#pragma once
#include <vector>
#include <deque>
#include <unordered_set>
#include <memory>
#include <thread>
#include <condition_variable>
#include <opencv2/opencv.hpp>
#include <ros/ros.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/StereoFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include "basic_datatype/frame.h"
#include "camera_model/simple_stereo_camera.h"

class ISAM2BackEnd {
public:
    enum BackEndState {
        INIT,
        NON_LINEAR
    };

    ISAM2BackEnd(const SimpleStereoCamPtr& camera);
    ~ISAM2BackEnd();

    void Process();
    void AddKeyFrame(const FramePtr& keyframe);

private:
    bool InitSystem(const FramePtr& keyframe);
    void CreateMapPointFromStereoMatching(const FramePtr& keyframe);
    void CreateFactorGraph(const FramePtr& keyframe);
    bool SolvePnP(const FramePtr& keyframe);

    BackEndState mState;
    SimpleStereoCamPtr mpCamera;
    gtsam::Cal3_S2::shared_ptr mKMono;
    gtsam::Cal3_S2Stereo::shared_ptr mKStereo;
    gtsam::ISAM2 mISAM;

    // buffer
    std::deque<FramePtr> mKFBuffer;
    std::mutex mKFBufferMutex;
    std::condition_variable mKFBufferCV;

    // store the feature id which has been triangulated.
    std::unordered_set<uint64_t> msIsTriangulate;

    gtsam::NonlinearFactorGraph mGraph;
    gtsam::Values mInitValues;
};

using ISAM2BackEndPtr = std::shared_ptr<ISAM2BackEnd>;
using ISAM2BackEndConstPtr = std::shared_ptr<const ISAM2BackEnd>;
