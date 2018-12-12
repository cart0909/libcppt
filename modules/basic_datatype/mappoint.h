#pragma once
#include <memory>
#include <mutex>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <deque>
#include "util_datatype.h"
#include "frame.h"

class Frame;
SMART_PTR(Frame)

class MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint();
    MapPoint(const FramePtr& keyframe, size_t idx);
    ~MapPoint();

    const uint64_t mID;
    Eigen::Vector3d x3Dw() const;
    Eigen::Vector3d x3Dc(const Sophus::SE3d& Tcw) const;
    void Set_x3Dw(const Eigen::Vector3d& x3Dw);
    void Set_x3Dw(const Eigen::Vector3d& x3Dc, const Sophus::SE3d& Twc);

    bool empty() const;
    void reset();

    // for mono feature create stereo
    void AddMeas(const FramePtr& keyframe, size_t idx);
    std::vector<std::pair<FramePtr, size_t>> GetMeas();

    // help us to traversal graph
    static uint64_t gTraversalId;
    uint64_t mTraversalId;
    uint64_t mVectorIdx;

    // ceres
    double vertex_data[3];
private:
    static uint64_t gNextID;
    bool mbNeedInit;
    Eigen::Vector3d m_x3Dw;
    mutable std::mutex mMutex;

    // these data will using to triangulate to 3d mappoint
    std::deque<std::pair<FrameWPtr, size_t>> mvMeas;
    mutable std::mutex mMeasMutex;
};

SMART_PTR(MapPoint)

// TODO
// bool isbad;
// size_t index;
// double inv_depth;
// FramePtr ref_frame;
