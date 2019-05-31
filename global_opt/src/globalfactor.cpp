#include "globalfactor.h"


RelativeRTError::RelativeRTError(const Sophus::SE3d& iTj_m_, double t_var_, double q_var_)
    :iTj_m(iTj_m_), t_var(t_var_), q_var(q_var_){}

ceres::CostFunction* RelativeRTError::Create(const Sophus::SE3d &iTj_m, double t_var,  double q_var)
{
    return (new ceres::AutoDiffCostFunction<RelativeRTError, 6, 3, 4, 3, 4>(new RelativeRTError(iTj_m, t_var, q_var)));
}


TError::TError(const double t_x, const double t_y, const double t_z, const double var):t_x(t_x), t_y(t_y), t_z(t_z), var(var){}

ceres::CostFunction* TError::Create( const double t_x,  const double t_y,  const double t_z,  const double var)
{
    return (new ceres::AutoDiffCostFunction<TError, 3, 3>(new TError(t_x, t_y, t_z, var)));
}
