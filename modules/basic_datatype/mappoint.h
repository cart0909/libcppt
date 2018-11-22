#pragma once
#include <memory>
#include <mutex>
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include "util_datatype.h"

class MapPoint {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    MapPoint(const Eigen::Vector3d& x3Dw);
    ~MapPoint();

    uint64_t mID;
    static uint64_t gNextID;

    Eigen::Vector3d x3Dw() const;
    Eigen::Vector3d x3Dc(const Sophus::SE3d& Tcw) const;
    void Set_x3Dw(const Eigen::Vector3d& x3Dw);
    void Set_x3Dw(const Eigen::Vector3d& x3Dc, const Sophus::SE3d& Twc);
private:
    Eigen::Vector3d m_x3Dw;
    mutable std::mutex mMutex;
};

SMART_PTR(MapPoint)
