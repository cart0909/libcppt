#pragma once
#include <memory>
#include <mutex>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "util_datatype.h"

class MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint();
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
private:
    bool mbNeedInit;
    Eigen::Vector3d m_x3Dw;

    mutable std::mutex mMutex;
    static uint64_t gNextID;
};

SMART_PTR(MapPoint)

// TODO
// bool isbad;
// size_t index;
// double inv_depth;
// FramePtr ref_frame;
