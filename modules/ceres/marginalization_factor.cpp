#include "marginalization_factor.h"

bool MarginalizationFactor::Evaluate(double const *const *parameters,
                                     double* residual, double **jacobian) const {
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
    for(int i = 0, n = keep_block.size(); i < n; ++i) {
        int size = keep_block[i]->size; // global size
        int idx = keep_block[i]->idx;
        MarginalizationInfo::VertexType vertex_type = keep_block[i]->vertex_type;
        Eigen::VectorXd x = Eigen::Map<const Eigen::VectorXd>(parameters[i], size);
        Eigen::VectorXd x0 = Eigen::Map<const Eigen::VectorXd>(keep_block[i]->data, size);

        if(vertex_type == MarginalizationInfo::VERTEX_SE3) {
            // Tcw
            // delT = Tc_c0 = Tcw * Tc0w.inv();

            // Twb
            // delT = Tb0_b = Twb0.inv() * Twb;
        }
        else {
            dx.segment(idx, size) = x - x0;
        }

    }

    return true;
}
