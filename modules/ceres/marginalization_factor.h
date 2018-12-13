#pragma once
// this code is reference from vins marginalization
#include <sophus/se3.hpp>
#include <ceres/ceres.h>
#include "basic_datatype/util_datatype.h"

class ParameterBlockInfo {
public:
    ParameterBlockInfo(int size_, int local_size_)
        : size(size_), local_size(local_size_), idx(-1), data(nullptr) {}
    ParameterBlockInfo(int size_)
        : ParameterBlockInfo(size_, size_) {}
    virtual void linearOMinus(const double* x, const double* x0, double* delx) const;

    const int size;
    const int local_size;
    int idx;
    std::unique_ptr<double[]> data;
};
SMART_PTR(ParameterBlockInfo)

class SE3BlockInfo : public ParameterBlockInfo {
public:
    SE3BlockInfo()
        : ParameterBlockInfo(Sophus::SE3d::num_parameters, Sophus::SE3d::DoF) {}
    void linearOMinus(const double* x, const double* x0, double* delx) const override;
};
SMART_PTR(SE3BlockInfo)

class ResidualBlockInfo {
public:
    ResidualBlockInfo(std::shared_ptr<ceres::CostFunction> cost_function_,
                      std::shared_ptr<ceres::LossFunction> loss_function_,
                      const std::vector<double *>& parameter_blocks_,
                      const std::vector<int>& drop_set_)
        : cost_function(cost_function_), loss_function(loss_function_), parameter_blocks(parameter_blocks_),
          drop_set(drop_set_) {}
    ~ResidualBlockInfo() {}

    void Evaluate();

    std::shared_ptr<ceres::CostFunction> cost_function;
    std::shared_ptr<ceres::LossFunction> loss_function;
    std::vector<double*>    parameter_blocks;
    std::vector<ParameterBlockInfoPtr> parameter_block_info;
    std::vector<int>        drop_set;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> jacobians;
    Eigen::VectorXd residuals;
};
SMART_PTR(ResidualBlockInfo)

class MarginalizationInfo {
public:
    void AddResidualBlockInfo(ResidualBlockInfoPtr residual_block_info);
    void AddParameterBlockInfo(double* vertex_data, ParameterBlockInfoPtr parameter_block_info);
    void PreMarginalize();
    void Marginalize();
    std::vector<double*> GetParameterBlocks();

    std::vector<ResidualBlockInfoPtr> factors;
    int m, n;
    int sum_block_size;
    std::map<double*, ParameterBlockInfoPtr> m_parameter_block_info;
    std::vector<ParameterBlockInfoPtr> keep_block;

    Eigen::MatrixXd linearized_jacobian;
    Eigen::VectorXd linearized_residuals;
    const double eps = 1e-8;
};
SMART_PTR(MarginalizationInfo)

// prior
class MarginalizationFactor : public ceres::CostFunction {
public:
    MarginalizationFactor(const MarginalizationInfoPtr& marginalization_info_);
    virtual bool Evaluate(double const *const *parameters_raw, double* residual_raw,
                          double **jacobian_raw) const;

    MarginalizationInfoWPtr marginalization_info;
};
