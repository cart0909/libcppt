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

    void reset();
    bool is_init() const;
    bool is_bad() const;
    void set_bad();

    Eigen::Vector3d x3Dw();
    Eigen::Vector3d x3Dc(const Sophus::SE3d& Tcw);
    // setter
    double inv_z() const;
    void inv_z(double inv_depth_);

    void add_meas(FramePtr kf, size_t idx);
    std::vector<std::pair<FramePtr, size_t>> get_meas();
    // temp function
    std::vector<std::pair<FramePtr, size_t>> get_meas(FramePtr kf);

    std::pair<FramePtr, size_t> get_parent();

    // ceres
    double vertex_data[1];
private:
    mutable std::mutex m_dep;
    bool b_init;
    bool b_bad;
    double inv_depth;
    Eigen::Vector3d x3Dw_; // cache

    std::mutex m_meas;
    std::deque<std::pair<FrameWPtr, size_t>> q_meas;
    FrameWPtr parent_kf;
    size_t parent_idx;
};
SMART_PTR(MapPoint)
