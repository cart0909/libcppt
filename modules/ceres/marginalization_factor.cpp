#include "marginalization_factor.h"
#include <sophus/se3.hpp>

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
    int n = marginalization_info->n;
    int m = marginalization_info->m;
    Eigen::VectorXd dx(n);

    auto& keep_block = marginalization_info->keep_block;
    for(int i = 0; i < static_cast<int>(keep_block.size()); ++i) {
        int size = keep_block[i]->size; // global size
        int idx = keep_block[i]->idx - m;
        ResidualBlockInfo::VertexType vertex_type = keep_block[i]->vertex_type;

        if(vertex_type == ResidualBlockInfo::VERTEX_SE3_TCW) {
            // delT = Tc_c0 = Tcw * Tc0w.inv();
            Eigen::Map<const Sophus::SE3d> Tcw(parameters_raw[i]);
            Eigen::Map<const Sophus::SE3d> Tc0w(keep_block[i]->data);
            Eigen::Map<Sophus::SE3d> delT(dx.data() + idx);
            delT = Tcw * Tc0w.inverse();
        }
        else if(vertex_type == ResidualBlockInfo::VERTEX_SE3_TWB) {
            // delT = Tb0_b = Twb0.inv() * Twb;
            Eigen::Map<const Sophus::SE3d> Twb(parameters_raw[i]);
            Eigen::Map<const Sophus::SE3d> Twb0(keep_block[i]->data);
            Eigen::Map<Sophus::SE3d> delT(dx.data() + idx);
            delT = Twb0.inverse() * Twb;
        }
        else {
            Eigen::Map<const Eigen::VectorXd> x(parameters_raw[i], size);
            Eigen::Map<const Eigen::VectorXd> x0(keep_block[i]->data, size);
            dx.segment(idx, size) = x - x0;
        }
    }

    Eigen::Map<Eigen::VectorXd> residuals(residual_raw, n);
    residuals = marginalization_info->linearized_residuals +
            marginalization_info->linearized_jacobian * dx;

    if(jacobian_raw) {
        for(int i = 0; i < static_cast<int>(marginalization_info->keep_block.size()); ++i) {
            int size = marginalization_info->keep_block[i]->size;
            int idx = marginalization_info->keep_block[i]->idx - m;
            ResidualBlockInfo::VertexType vertex_type = keep_block[i]->vertex_type;
            int local_size = size;
            if(vertex_type == ResidualBlockInfo::VERTEX_SE3_TCW ||
                    vertex_type == ResidualBlockInfo::VERTEX_SE3_TWB)
                local_size = Sophus::SE3d::DoF;

            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                    Eigen::RowMajor>> jacobian(jacobian_raw[i], n, size);
            jacobian.setZero();
            jacobian.leftCols(local_size) = marginalization_info->linearized_jacobian.middleCols(idx, local_size);
        }
    }
    return true;
}
