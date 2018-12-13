#include "marginalization_factor.h"
#include <tracer.h>
#include <ros/ros.h>

void ParameterBlockInfo::linearOMinus(const double* x_raw, const double* x0_raw, double* delx_raw) const {
    Eigen::Map<const Eigen::VectorXd> x(x_raw, size), x0(x0_raw, size);
    Eigen::Map<Eigen::VectorXd> delx(delx_raw, size);
    delx = x - x0;
}

void SE3BlockInfo::linearOMinus(const double* x_raw, const double* x0_raw, double* delx_raw) const {
    // example, Twc, Twc0
    // Twc = Twc0 * delT
    // delT = Tc0c = Twc0.inv() * Twc
    Eigen::Map<const Sophus::SE3d> T(x_raw), T0(x0_raw);
    Eigen::Map<Sophus::SE3d> delT(delx_raw);
    delT = T0.inverse() * T;
}

void ResidualBlockInfo::Evaluate() {
    residuals.resize(cost_function->num_residuals());

    std::vector<int> block_sizes = cost_function->parameter_block_sizes();
    std::vector<double*> tmp_jacobians(block_sizes.size());
    jacobians.resize(block_sizes.size());
    for(int i = 0, n = block_sizes.size(); i < n; ++i) {
        jacobians[i].resize(cost_function->num_residuals(), block_sizes[i]);
        tmp_jacobians[i] = jacobians[i].data();
    }

    cost_function->Evaluate(parameter_blocks.data(), residuals.data(), tmp_jacobians.data());

    // from ceres source code
    if(loss_function) {
        double residual_scaling_, alpha_sq_norm_;
        double sq_norm, rho[3];

        sq_norm = residuals.squaredNorm();
        loss_function->Evaluate(sq_norm, rho);

        double sqrt_rho1_ = sqrt(rho[1]);

        if ((sq_norm == 0.0) || (rho[2] <= 0.0))
        {
            residual_scaling_ = sqrt_rho1_;
            alpha_sq_norm_ = 0.0;
        }
        else
        {
            const double D = 1.0 + 2.0 * sq_norm * rho[2] / rho[1];
            const double alpha = 1.0 - sqrt(D);
            residual_scaling_ = sqrt_rho1_ / (1 - alpha);
            alpha_sq_norm_ = alpha / sq_norm;
        }

        for (int i = 0; i < static_cast<int>(parameter_blocks.size()); i++)
        {
            jacobians[i] = sqrt_rho1_ * (jacobians[i] - alpha_sq_norm_ * residuals * (residuals.transpose() * jacobians[i]));
        }

        residuals *= residual_scaling_;
    }
}

void MarginalizationInfo::AddResidualBlockInfo(ResidualBlockInfoPtr residual_block_info) {
    factors.emplace_back(residual_block_info);

    auto& parameter_blocks = residual_block_info->parameter_blocks;
    auto& parameter_block_info = residual_block_info->parameter_block_info; // size 0
    auto parameter_block_sizes = residual_block_info->cost_function->parameter_block_sizes();

    for(int i = 0, n = parameter_blocks.size(); i < n; ++i) {
        double* addr = parameter_blocks[i];
        auto it = m_parameter_block_info.find(addr);
        if(it == m_parameter_block_info.end()) {
            auto temp = std::make_shared<ParameterBlockInfo>(parameter_block_sizes[i]);
            m_parameter_block_info[addr] = temp;
            parameter_block_info.emplace_back(temp);
            ROS_ERROR_STREAM("auto create parameter block: " << temp.get());
        }
        else {
            parameter_block_info.emplace_back(it->second);
        }
    }

    for(auto& idx : residual_block_info->drop_set) {
        if(parameter_block_info[idx]->idx != 0)
            parameter_block_info[idx]->idx = 0;
    }
}

void MarginalizationInfo::AddParameterBlockInfo(double* vertex_data, ParameterBlockInfoPtr parameter_block_) {
    m_parameter_block_info[vertex_data] = parameter_block_;
}

void MarginalizationInfo::PreMarginalize() {
    ScopedTrace st("pre_margin");
    for(auto& it : factors) {
        it->Evaluate();

        std::vector<int> block_sizes = it->cost_function->parameter_block_sizes();
        for(int i = 0, n = block_sizes.size(); i < n; ++i) {
            int size = block_sizes[i];
            if(!it->parameter_block_info[i]->data) {
                it->parameter_block_info[i]->data = std::unique_ptr<double[]>(new double[size]);
                std::memcpy(it->parameter_block_info[i]->data.get(),
                            it->parameter_blocks[i], sizeof(double) * size);
            }
        }
    }
}

void MarginalizationInfo::Marginalize() {
    ScopedTrace st("margin");
    m = 0; // marginal size
    n = 0; // non marginal size
    for(auto& it : m_parameter_block_info) {
        auto& block = it.second;
        if(block->idx == 0) { // marginal
            block->idx = m;
            m += block->local_size;
        }
    }

    for(auto& it : m_parameter_block_info) {
        auto& block = it.second;
        if(block->idx == -1) { // non marginal
            block->idx = m + n;
            n += block->local_size;
        }
    }
    int total_size = m + n;
    ROS_ERROR_STREAM("m " << m << " n " << n);
    /*
     * A = [Amm Amr   b = [bmm
     *      Arm Arr]       brr]
     */

    Eigen::MatrixXd A(total_size, total_size);
    Eigen::VectorXd b(total_size);
    A.setZero();
    b.setZero();

    Tracer::TraceBegin("fillA");
    // TODO fill the matrix A and vector b using multi-thread
    for(auto& factor : factors) {
        for(int i = 0; i < factor->parameter_blocks.size(); ++i) {
            ROS_ERROR_STREAM("i:" << i << " size:" << factor->parameter_blocks.size());
            int idx_i = factor->parameter_block_info[i]->idx;
            int local_size_i = factor->parameter_block_info[i]->local_size;
            ROS_ERROR_STREAM("idx_i:" << idx_i << " local_size_i:" << local_size_i);
            if(idx_i == -1) {
                auto it = m_parameter_block_info.find(factor->parameter_blocks[i]);
                if(it != m_parameter_block_info.end()) {
                    ROS_ERROR_STREAM(factor->parameter_block_info[i].get() << " " << it->second.get());
                }
                else {
                    ROS_ERROR_STREAM("nofind");
                }
            }
            Eigen::MatrixXd jacobian_i = factor->jacobians[i].leftCols(local_size_i);

            // fill bi
            b.segment(idx_i, local_size_i) = jacobian_i.transpose() * factor->residuals;

            // fill Aij
            for(int j = i; j < factor->parameter_blocks.size(); ++j) {
                int idx_j = factor->parameter_block_info[j]->idx;
                int local_size_j = factor->parameter_block_info[j]->local_size;
                Eigen::MatrixXd jacobian_j = factor->jacobians[j].leftCols(local_size_j);
                A.block(idx_i, idx_j, local_size_i, local_size_j) += jacobian_i.transpose() * jacobian_j;
                if(i != j)
                    A.block(idx_j, idx_i, local_size_j, local_size_i) =
                            A.block(idx_i, idx_j, local_size_i, local_size_j).transpose();
            }
        }
    }
    Tracer::TraceEnd();
    /*
     * Solve Ax = b and Amn = Anm^t
     *
     *      [Amm Amn [xm  = [bm
     *       Anm Ann] xn]    bn]
     *
     * Using Schur complement
     *      [ Amm                   Amn  [xm   = [                  bm
     *          0   Ann-Anm*Amm_inv*Amn]  xn]     bn - Anm*Amm_inv*Amn]
     *    =>(Ann-Anm*Amm_inv*Amn)xn = bn - Anm*Amm_inv*bm
     */
    Eigen::MatrixXd Amm = 0.5 * (A.block(0, 0, m, m) + A.block(0, 0, m, m).transpose()); // ensure the Amm is symmetry
    // https://eigen.tuxfamily.org/dox/classEigen_1_1SelfAdjointEigenSolver.html
    Tracer::TraceBegin("VDVt");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes(Amm); // Amm = VDV^t
    Tracer::TraceEnd();
    Tracer::TraceBegin("shur");
    // Amm_inv = VD^-1V^t
    Eigen::VectorXd D_inv = Eigen::VectorXd((saes.eigenvalues().array() > eps).select(saes.eigenvalues().array().inverse(), 0));
    Eigen::MatrixXd Amm_inv = saes.eigenvectors() * D_inv.asDiagonal() * saes.eigenvectors().transpose();

    Eigen::MatrixXd Ann = A.block(m, m, n, n);
    Eigen::MatrixXd Anm = A.block(m, 0, n, m);
    Eigen::MatrixXd Amn = A.block(0, m, m, n);
    Eigen::VectorXd bm = b.segment(0, m);
    Eigen::VectorXd bn = b.segment(m, n);

    A = Ann - Anm * Amm_inv * Amn;
    b = bn - Anm * Amm_inv * bm;
    Tracer::TraceEnd();
    // Ax = b
    // J^tJ x = J^te
    // try to find J
    // decomposition A = VSV^t = J^tJ
    // then J = sqrt(D)*V^t
    //      e = (sqrt(D))^-1 * V^t * b

    Tracer::TraceBegin("VDVt");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> saes2(A);
    Tracer::TraceEnd();
    Eigen::VectorXd D = (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array(), 0);
    D_inv = (saes2.eigenvalues().array() > eps).select(saes2.eigenvalues().array().inverse(), 0);
    Eigen::VectorXd D_sqrt = D.cwiseSqrt();
    Eigen::VectorXd D_inv_sqrt = D_inv.cwiseSqrt();

    linearized_jacobian = D_sqrt.asDiagonal() * saes2.eigenvectors().transpose();
    linearized_residuals = D_inv_sqrt.asDiagonal() * saes2.eigenvectors().transpose() * b;
}

std::vector<double*> MarginalizationInfo::GetParameterBlocks() {
    std::vector<double*> keep_block_addr;
    keep_block.clear();
    for(auto& it : m_parameter_block_info) {
        if(it.second->idx >= m) {
            keep_block.emplace_back(it.second);
            keep_block_addr.emplace_back(it.first);
        }
    }
    // keep_block_addr is correspondence to keep_block
    return keep_block_addr;
}

MarginalizationFactor::MarginalizationFactor(const MarginalizationInfoPtr& marginalization_info_)
    : marginalization_info(marginalization_info_)
{
    for(auto& it : marginalization_info_->keep_block) {
        mutable_parameter_block_sizes()->push_back(it->size);
    }
    set_num_residuals(marginalization_info_->n);
}

bool MarginalizationFactor::Evaluate(double const *const *parameters_raw,
                                     double* residual_raw, double **jacobian_raw) const {
    /*
     *  <----m----> <-----n----->
     *  0 ... (m-1) m ... (m+n-1)
     * [           |               ^
     *             |               | m
     *             |               v
     *  -------------------------  ^
     *             |               | n
     *             |               |
     *             |             ] v
     */
    if(marginalization_info.expired())
        return false;
    auto info = marginalization_info.lock();
    int n = info->n;
    int m = info->m;
    Eigen::VectorXd dx(n);

    auto& keep_block = info->keep_block;
    for(int i = 0; i < static_cast<int>(keep_block.size()); ++i) {
        int size = keep_block[i]->size; // global size
        int idx = keep_block[i]->idx - m;
        keep_block[i]->linearOMinus(parameters_raw[i], keep_block[i]->data.get(), dx.data() + idx);
    }

    Eigen::Map<Eigen::VectorXd> residuals(residual_raw, n);
    residuals = info->linearized_residuals + info->linearized_jacobian * dx;

    if(jacobian_raw) {
        for(int i = 0; i < static_cast<int>(info->keep_block.size()); ++i) {
            int size = info->keep_block[i]->size;
            int local_size = info->keep_block[i]->local_size;
            int idx = info->keep_block[i]->idx - m;
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>> jacobian(jacobian_raw[i], n, size);
            jacobian.setZero();
            jacobian.leftCols(local_size) = info->linearized_jacobian.middleCols(idx, local_size);
        }
    }
    return true;
}
