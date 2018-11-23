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
    MapPoint(const FrameConstPtr& keyframe, const cv::Point2f& uv);
    ~MapPoint();

    const uint64_t mID;
    Eigen::Vector3d x3Dw() const;
    Eigen::Vector3d x3Dc(const Sophus::SE3d& Tcw) const;
    void Set_x3Dw(const Eigen::Vector3d& x3Dw);
    void Set_x3Dw(const Eigen::Vector3d& x3Dc, const Sophus::SE3d& Twc);

    // x3Dc = Tcw * x3Dw
    // d(x3Dc)/d(Xi) = [I, -x3Dc^]
    Eigen::Matrix<double, 3, 6> Jacobian(const Sophus::SE3d& Tcw) const;

    bool empty() const;
    void reset();

    void AddMeas(const FrameConstPtr& keyframe, const cv::Point2f& uv);
    std::vector<std::pair<FrameConstPtr, cv::Point2f>> GetMeas();

    // help us to traversal graph
    static uint64_t gTraversalId;
    uint64_t mTraversalId;
private:
    static uint64_t gNextID;
    bool mbNeedInit;
    Eigen::Vector3d m_x3Dw;
    mutable std::mutex mMutex;

    // these data will using to triangulate to 3d mappoint
    std::deque<std::pair<FrameConstWPtr, cv::Point2f>> mvMeas;
    mutable std::mutex mMeasMutex;
};

SMART_PTR(MapPoint)

// TODO
// bool isbad;
// size_t index;
// double inv_depth;
// FramePtr ref_frame;
