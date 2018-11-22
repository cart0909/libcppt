#include "mappoint.h"

uint64_t MapPoint::gNextID = 0;

MapPoint::MapPoint(const Eigen::Vector3d& x3Dw)
    : mID(gNextID++), m_x3Dw(x3Dw)
{

}

MapPoint::~MapPoint() {

}

Eigen::Vector3d MapPoint::x3Dw() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return m_x3Dw;
}

Eigen::Vector3d MapPoint::x3Dc(const Sophus::SE3d& Tcw) const {
    mMutex.lock();
    Eigen::Vector3d x3Dw = m_x3Dw;
    mMutex.unlock();
    return Tcw * x3Dw;
}

void MapPoint::Set_x3Dw(const Eigen::Vector3d& x3Dw) {
    std::unique_lock<std::mutex> lock(mMutex);
    m_x3Dw = x3Dw;
}

void MapPoint::Set_x3Dw(const Eigen::Vector3d& x3Dc, const Sophus::SE3d& Twc) {
    Eigen::Vector3d x3Dw = Twc * x3Dc;
    std::unique_lock<std::mutex> lock(mMutex);
    m_x3Dw = x3Dw;
}
