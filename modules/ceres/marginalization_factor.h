#pragma once
// this code is reference from vins marginalization
#include <ceres/ceres.h>
#include "basic_datatype/util_datatype.h"

class ResidualBlockInfo {
public:
    ResidualBlockInfo(ceres::CostFunction* cost_function_, ceres::LossFunction* loss_function_,
                      const std::vector<double *>& parameter_blocks_, const std::vector<int>& drop_set_)
        : cost_function(cost_function_), loss_function(loss_function_), parameter_blocks(parameter_blocks_),
          drop_set(drop_set_) {}
    ~ResidualBlockInfo() {}

    void Evaluate();

    ceres::CostFunction* cost_function;
    ceres::LossFunction* loss_function;
    std::vector<double*> parameter_blocks; // double**
    std::vector<int>     drop_set;
    double **raw_jacobians;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
};

SMART_PTR(ResidualBlockInfo)

class MarginalizationInfo {
public:
    enum VertexType {
        VERTEX_LINEAR,
        VERTEX_SE3
    };
    // ????
    struct ParameterBlock {     // example SE3
        VertexType vertex_type;
        int size; // global size   7
        int idx;  // local size    6
        double* data;
    };
    SMART_PTR(ParameterBlock)
    // ????

    MarginalizationInfo() {}
    ~MarginalizationInfo() {}
    std::vector<ResidualBlockInfoPtr> factors;
    int m, n;
    int sum_block_size;
    std::map<long, int>     parameter_block_size; // global size
    std::map<long, int>     parameter_block_idx;  // local size
    std::map<long, double*> parameter_block_data;
    std::map<long, ParameterBlockPtr> parameter_block; // carl?

    std::vector<VertexType> keep_block_vertextype;
    std::vector<int>        keep_block_size; // global size
    std::vector<int>        keep_block_idx;  // local size
    std::vector<double*>    keep_block_data;
    std::vector<ParameterBlockPtr> keep_block; // carl?

    Eigen::MatrixXd linearized_jacobian;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};

SMART_PTR(MarginalizationInfo)

// prior
class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(const MarginalizationInfoPtr& marginalization_info_)
        : marginalization_info(marginalization_info_) {}
    virtual bool Evaluate(double const *const *parameters, double* residual, double **jacobian) const;

    MarginalizationInfoPtr marginalization_info;
};
