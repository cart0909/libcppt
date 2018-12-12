#include "mappoint.h"

uint64_t MapPoint::gNextID = 0;
uint64_t MapPoint::gTraversalId = 0;

MapPoint::MapPoint()
    : mID(0), mbNeedInit(true)
{
}

MapPoint::MapPoint(const FramePtr& keyframe, size_t idx)
    : mID(0), mbNeedInit(true)
{
    AddMeas(keyframe, idx);
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

bool MapPoint::empty() const {
    std::unique_lock<std::mutex> lock(mMutex);
    return mbNeedInit;
}

void MapPoint::reset() {
    std::unique_lock<std::mutex> lock(mMutex);
    mbNeedInit = true;
}

void MapPoint::AddMeas(const FramePtr& keyframe, size_t idx) {
    std::unique_lock<std::mutex> lock(mMeasMutex);
    mvMeas.emplace_back(keyframe, idx);
}

std::vector<std::pair<FramePtr, size_t>> MapPoint::GetMeas() {
    std::vector<std::pair<FramePtr, size_t>> meas;
    std::unique_lock<std::mutex> lock(mMeasMutex);
    for(auto it = mvMeas.begin(); it != mvMeas.end();) {
        if(it->first.expired()) {
            it = mvMeas.erase(it);
        }
        else {
            meas.emplace_back(it->first.lock(), it->second);
            ++it;
        }
    }
    return meas;
}
