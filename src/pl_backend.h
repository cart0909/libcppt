#pragma once
#include "util.h"
#include "backend.h"

class PLBackEnd : public BackEnd {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    struct Frame : public BackEnd::Frame
    {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        virtual ~Frame() {}
        std::vector<uint64_t> line_id;
        Eigen::VecVector3d spt_n, ept_n; // spt: start point, ept: end point, n: normal plane
        Eigen::VecVector3d spt_r_n, ept_r_n; // r: right camera
    };
    SMART_PTR(Frame)

    struct LineFeature {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        uint64_t feat_id;
        int start_id;
        double inv_depth[2]; // [0]: spt inv_depth
                             // [1]: ept inv_depth
        Eigen::DeqVector3d spt_n_per_frame, ept_n_per_frame;
        Eigen::DeqVector3d spt_r_n_per_frame, ept_r_n_per_frame;
    };
    SMART_PTR(LineFeature)

private:
    // ceres data
    size_t  para_line_features_capacity = 1000;
    double* para_line_features;
};
