#pragma once
#include <memory>
#include <vector>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/nonlinear/Values.h>
#include <gtsam/nonlinear/NonlinearEquality.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/slam/StereoFactor.h>
#include "basic_datatype/frame.h"
#include "camera_model/simple_stereo_camera.h"

class ISAM2BackEnd {
public:
    ISAM2BackEnd(const SimpleStereoCamPtr& camera) {
        assert(camera != nullptr);
        mKStereo = boost::make_shared<gtsam::Cal3_S2Stereo>(camera->f, camera->f, 0,
                                                            camera->cx, camera->cy, camera->b);
        mKMono = boost::make_shared<gtsam::Cal3_S2>(camera->f, camera->f, 0, camera->cx, camera->cy);
        gtsam::ISAM2Params parameters;
        parameters.relinearizeThreshold = 0.01;
        parameters.relinearizeSkip = 1;
        mISAM = gtsam::ISAM2(parameters);
    }

    ~ISAM2BackEnd() {}
private:
    gtsam::Cal3_S2::shared_ptr mKMono;
    gtsam::Cal3_S2Stereo::shared_ptr mKStereo;
    gtsam::ISAM2 mISAM;
};

using ISAM2BackEndPtr = std::shared_ptr<ISAM2BackEnd>;
using ISAM2BackEndConstPtr = std::shared_ptr<const ISAM2BackEnd>;
