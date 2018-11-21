#include "isam2_backend.h"
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
//    parameters.relinearizeThreshold = 0.01;
//    parameters.relinearizeSkip = 1;
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
                    const Value& currentEstimate = mISAM.calculateEstimate(Symbol('x', 0));
                    const Pose3& T = currentEstimate.cast<Pose3>();
                    mGraph.resize(0);
                    mInitValues.clear();
                    mState = NON_LINEAR;
                }
            }
            else if(mState == NON_LINEAR) {
                // solve the linear system get init pose
                SolvePnP(keyframe);
                mInitValues.insert(Symbol('x', keyframe->mKeyFrameID),
                                   Pose3(keyframe->mTwc.matrix()));
                // create new map point
                CreateMapPointFromStereoMatching(keyframe);
                // create factor graph
                CreateFactorGraph(keyframe);

                Tracer::TraceBegin("isam2");
                mISAM.update(mGraph, mInitValues);
                Tracer::TraceEnd();
                auto& currentEstimate = mISAM.calculateEstimate(Symbol('x',keyframe->mKeyFrameID));
                const Pose3& T = currentEstimate.cast<Pose3>();

                std::cout << keyframe->mKeyFrameID <<" before\n";
                std::cout << keyframe->mTwc.matrix() << std::endl;
                std::cout << keyframe->mKeyFrameID<< " after\n";
                std::cout << T.matrix() << std::endl;
                mGraph.resize(0);
                mInitValues.clear();
            }
            else {
                ROS_ERROR_STREAM("BackEnd error state!");
                exit(-1);
            }
        }
    }
}

void ISAM2BackEnd::AddKeyFrame(const FramePtr& keyframe) {
    assert(keyframe->mIsKeyFrame);
    mKFBufferMutex.lock();
    mKFBuffer.push_back(keyframe);
    mKFBufferMutex.unlock();
    mKFBufferCV.notify_one();
}

void ISAM2BackEnd::CreateMapPointFromStereoMatching(const FramePtr& keyframe) {
    ScopedTrace st("CreateMP");
    auto& v_pt_id = keyframe->mvPtID;
    auto& v_uv = keyframe->mv_uv;
    auto& v_ur = keyframe->mv_ur;
    int N = v_uv.size();

//    static bool bfirst = true;

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

//        if(bfirst) {
//            // Add a prior on landmark l0
        static auto kPointPrior = noiseModel::Isotropic::Sigma(3, 0.1);
        mGraph.emplace_shared<PriorFactor<Point3>>(Symbol('l', v_pt_id[i]),
                                                   Point3(x3Dw(0), x3Dw(1), x3Dw(2)),
                                                   kPointPrior);
//            bfirst = false;
//        }
    }
}

bool ISAM2BackEnd::InitSystem(const FramePtr& keyframe) {
    if(keyframe->mNumStereo < 100)
        return false;

    CreateMapPointFromStereoMatching(keyframe);
    // add edge to graph
    mGraph.emplace_shared<NonlinearEquality<Pose3> >(Symbol('x', keyframe->mKeyFrameID), Pose3());
    mInitValues.insert(Symbol('x', keyframe->mKeyFrameID), Pose3());
    //create factor noise model with 3 sigmas of value 1
    const auto model = noiseModel::Isotropic::Sigma(3, 1);

    int N = keyframe->mv_uv.size();
    uint64_t keyframe_id = keyframe->mKeyFrameID;
    for(int i = 0; i < N; ++i) {
        if(keyframe->mv_ur[i] == -1)
            continue;
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

bool ISAM2BackEnd::SolvePnP(const FramePtr& keyframe) {
    ScopedTrace st("SolvePnP");
    auto& pt_id = keyframe->mvPtID;
    auto& pt = keyframe->mv_uv;
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
        auto it = msIsTriangulate.find(pt_id[i]);
        if(it != msIsTriangulate.end()) {
            auto& value = mISAM.calculateEstimate(Symbol('l', pt_id[i]));
            auto& x3Dw = value.cast<Point3>();
            object_points.emplace_back(x3Dw.x(), x3Dw.y(), x3Dw.z());
            image_points.emplace_back(pt[i].x, pt[i].y);
        }
    }

    if(image_points.size() > 16) {
        cv::Mat rvec, tvec, R;
        cv::solvePnP(object_points, image_points, K, cv::noArray(), rvec, tvec, false,
                     cv::SOLVEPNP_EPNP);
        cv::Rodrigues(rvec, R);
        Eigen::Map<Eigen::Matrix<double, 3, 3, Eigen::RowMajor>> eigen_R(R.ptr<double>());
        Eigen::Map<Eigen::Vector3d> eigen_t(tvec.ptr<double>());
        Sophus::SE3d Tcw(eigen_R, eigen_t), Twc = Tcw.inverse();
        keyframe->mTwc = Twc;
        return true;
    }
    return false;
}

void ISAM2BackEnd::CreateFactorGraph(const FramePtr& keyframe) {
    ScopedTrace st("CreateFactorGraph");
    auto& pt_id = keyframe->mvPtID;
    auto& pt_uv = keyframe->mv_uv;
    auto& pt_ur = keyframe->mv_ur;
    uint64_t keyframe_id = keyframe->mKeyFrameID;
    int N = keyframe->mv_uv.size();

    //create factor noise model with 3 sigmas of value 1
    const auto model2 = noiseModel::Isotropic::Sigma(2, 1);
    const auto model3 = noiseModel::Isotropic::Sigma(3, 1);

    for(int i = 0; i < N; ++i) {
        auto it = msIsTriangulate.find(pt_id[i]);
        if(it == msIsTriangulate.end())
            continue;
        uint64_t landmark_id = pt_id[i];
        double u = pt_uv[i].x;
        double v = pt_uv[i].y;
        double ur = pt_ur[i];

        // mono edge
        if(ur == -1) {
            mGraph.emplace_shared<GenericProjectionFactor<Pose3, Point3>>
                    (Point2(u, v), model2,
                     Symbol('x', keyframe_id),
                     Symbol('l', landmark_id),
                     mKMono);
        }
        else { // stereo edge
            mGraph.emplace_shared<GenericStereoFactor<Pose3, Point3>>
                    (StereoPoint2(u, ur, v), model3,
                    Symbol('x', keyframe_id),
                    Symbol('l', landmark_id),
                    mKStereo);
        }
    }
}
