#include "mappoint.h"

MapPoint::MapPoint()
    : vertex_data{0}, b_init(false), b_bad(false), inv_depth(-1)
{
}

void MapPoint::reset() {
    std::unique_lock<std::mutex> lock(m_dep), lock1(m_meas);
    b_init = false;
    b_bad = false;
    inv_depth = -1;
    q_meas.clear();
}

bool MapPoint::is_init() const {
    std::unique_lock<std::mutex> lock(m_dep);
    return b_init;
}

bool MapPoint::is_bad() const {
    std::unique_lock<std::mutex> lock(m_dep);
    return b_bad;
}

void MapPoint::set_bad() {
    std::unique_lock<std::mutex> lock(m_dep);
    if(!b_bad)
        b_bad = true;
}

Eigen::Vector3d MapPoint::x3Dw() {
    std::unique_lock<std::mutex> lock(m_dep);
    return x3Dw_;
}

Eigen::Vector3d MapPoint::x3Dc(const Sophus::SE3d& Tcw) {
    std::unique_lock<std::mutex> lock(m_dep);
    return Tcw * x3Dw_;
}

double MapPoint::inv_z() const {
    std::unique_lock<std::mutex> lock(m_dep);
    return inv_depth;
}

void MapPoint::inv_z(double inv_depth_) {
    std::unique_lock<std::mutex> lock(m_dep);
    inv_depth = inv_depth_;
    if(!b_init)
        b_init = true;

    auto kf = parent_kf.lock();
    double inv_f = kf->mpCamera->inv_f;
    double cx = kf->mpCamera->cx;
    double cy = kf->mpCamera->cy;
    cv::Point2f uv = kf->mv_uv[parent_idx];
    double z = 1.0 / inv_depth;
    Eigen::Vector3d x3Dc;
    x3Dc << z * inv_f * (uv.x - cx),
            z * inv_f * (uv.y - cy),
            z;

    Sophus::SE3d& Twc = kf->mTwc;
    x3Dw_ = Twc * x3Dc;
}

void MapPoint::add_meas(FramePtr kf, size_t idx) {
    std::unique_lock<std::mutex> lock(m_dep);
    if(q_meas.empty()) {
        parent_kf = kf;
        parent_idx = idx;
    }
    q_meas.emplace_back(kf, idx);
}

std::vector<std::pair<FramePtr, size_t>> MapPoint::get_meas() {
    std::vector<std::pair<FramePtr, size_t>> temp;
    std::unique_lock<std::mutex> lock(m_dep);
    for(auto it = q_meas.begin(); it != q_meas.end();) {
        if(it->first.expired()) {
            it = q_meas.erase(it);
        }
        else {
            temp.emplace_back(it->first.lock(), it->second);
            ++it;
        }
    }
    return temp;
}

std::vector<std::pair<FramePtr, size_t>> MapPoint::get_meas(FramePtr kf) {
    std::vector<std::pair<FramePtr, size_t>> temp;
    std::unique_lock<std::mutex> lock(m_dep);
    for(auto it = q_meas.begin(); it != q_meas.end();) {
        if(it->first.expired()) {
            it = q_meas.erase(it);
        }
        else {
            if(it->first.lock()->mKeyFrameID <= kf->mKeyFrameID)
                temp.emplace_back(it->first.lock(), it->second);
            ++it;
        }
    }
    return temp;
}

std::pair<FramePtr, size_t> MapPoint::get_parent() {
    std::unique_lock<std::mutex> lock(m_dep);
    return std::make_pair(parent_kf.lock(), parent_idx);
}
