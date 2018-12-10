#pragma once
// this code is reference from vins marginalization
#include <ceres/ceres.h>
#include "basic_datatype/util_datatype.h"

class ResidualBlockInfo {
public:

    enum VertexType {
        VERTEX_LINEAR,
        VERTEX_SE3_TCW,
        VERTEX_SE3_TWB
    };

    ResidualBlockInfo(std::shared_ptr<ceres::CostFunction> cost_function_,
                      std::shared_ptr<ceres::LossFunction> loss_function_,
                      const std::vector<double *>& parameter_blocks_,
                      const std::vector<VertexType>& vertex_types_,
                      const std::vector<int>& drop_set_)
        : cost_function(cost_function_), loss_function(loss_function_), parameter_blocks(parameter_blocks_),
          vertex_types(vertex_types_), drop_set(drop_set_) {}
    ~ResidualBlockInfo() {}

    void Evaluate();

    std::shared_ptr<ceres::CostFunction> cost_function;
    std::shared_ptr<ceres::LossFunction> loss_function;
    std::vector<double*>    parameter_blocks; // double**
    std::vector<VertexType> vertex_types;
    std::vector<int>        drop_set;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
};

SMART_PTR(ResidualBlockInfo)

class MarginalizationInfo {
public:
    struct ParameterBlock {
        ParameterBlock(ResidualBlockInfo::VertexType vertex_type_, int size_, int idx_, double* data_)
            : vertex_type(vertex_type_), size(size_), idx(idx_), data(data_) {}
        ResidualBlockInfo::VertexType vertex_type;
        int size;
        int idx;
        double* data;
    };
    SMART_PTR(ParameterBlock)

    void AddResidualBlockInfo(ResidualBlockInfoPtr residual_block_info);

    std::vector<ResidualBlockInfoPtr> factors;
    int m, n;
    int sum_block_size;
    std::map<double*, ParameterBlockPtr> parameter_block;
    std::vector<ParameterBlockPtr> keep_block;

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
    virtual bool Evaluate(double const *const *parameters_raw, double* residual_raw,
                          double **jacobian_raw) const;

    MarginalizationInfoPtr marginalization_info;
};
