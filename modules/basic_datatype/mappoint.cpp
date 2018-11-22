#include "mappoint.h"

uint64_t MapPoint::gNextID = 0;
uint64_t MapPoint::gTraversalId = 0;

MapPoint::MapPoint()
    : mID(0), mbNeedInit(true)
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
    if(mbNeedInit) {
        const_cast<uint64_t&>(mID) = gNextID++;
        mbNeedInit = false;
    }
}

void MapPoint::Set_x3Dw(const Eigen::Vector3d& x3Dc, const Sophus::SE3d& Twc) {
    Eigen::Vector3d x3Dw = Twc * x3Dc;
    std::unique_lock<std::mutex> lock(mMutex);
    m_x3Dw = x3Dw;
    if(mbNeedInit) {
        const_cast<uint64_t&>(mID) = gNextID++;
        mbNeedInit = false;
    }
}

Eigen::Matrix<double, 3, 6> MapPoint::Jacobian(const Sophus::SE3d& Tcw) const {
    Eigen::Matrix<double, 3, 6> J;
    mMutex.lock();
    Eigen::Vector3d x3Dw = m_x3Dw;
    mMutex.unlock();
    Eigen::Vector3d x3Dc = Tcw * x3Dw;
    J << Eigen::Matrix3d::Identity(), Sophus::SO3d::hat(x3Dc);
    return J;
}

bool MapPoint::empty() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return mbNeedInit;
}

void MapPoint::reset() {
    std::unique_lock<std::mutex> lock(mMutex);
    mbNeedInit = true;
}
