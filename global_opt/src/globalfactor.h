#pragma once
#include <ceres/ceres.h>
#include "ceres/local_parameterization_se3.h"
#include "util.h"
#include <ceres/autodiff_cost_function.h>

class TError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TError(const double t_x, const double t_y, const double t_z, const double var);


    template <typename T>
    bool operator()(const T* wvioTj_, T* residuals) const
    {
        Eigen::Map<const Sophus::SE3<T>> wvioTi(wvioTj_);
        residuals[0] = (wvioTi.translation().x() - T(t_x)) / T(var);
        residuals[1] = (wvioTi.translation().y() - T(t_y)) / T(var);
        residuals[2] = (wvioTi.translation().z() - T(t_z)) / T(var);

        return true;
    }

    static ceres::CostFunction* Create(const double t_x, const double t_y, const double t_z, const double var);


    double t_x, t_y, t_z, var;

};

class RelativeRTError
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RelativeRTError(const Sophus::SE3d& iTj_m_, double t_var_, double q_var_);
    template <class T>

    bool operator()(const T*  wvioPi_, const T*  wvioQi_, const T* wvioPj_, const T* wvioQj_,T* residuals) const
    {
        Eigen::Map<const Eigen::Quaternion<T>> wvioQi(wvioQi_);
        Eigen::Map<const Eigen::Quaternion<T>> wvioQj(wvioQj_);
        T t_w_ij[3];
        t_w_ij[0] = wvioPj_[0] - wvioPi_[0];
        t_w_ij[1] = wvioPj_[1] - wvioPi_[1];
        t_w_ij[2] = wvioPj_[2] - wvioPi_[2];

//        T t_i_ij[3];
//        t_i_ij = wvioQi.inverse() * t_w_ij;


//        residuals[0] = (t_i_ij[0] - iTj_m.translation().x()) / T(t_var);
//        residuals[1] = (t_i_ij[1] - iTj_m.translation().y()) / T(t_var);
//        residuals[2] = (t_i_ij[2] - iTj_m.translation().z()) / T(t_var);

//        T relative_q[4];
//        relative_q[0] = T(q_w);
//        relative_q[1] = T(q_x);
//        relative_q[2] = T(q_y);
//        relative_q[3] = T(q_z);

//        T q_i_j[4];
//        ceres::QuaternionProduct(i_q_w, w_q_j, q_i_j);

//        T relative_q_inv[4];
//        QuaternionInverse(relative_q, relative_q_inv);

//        T error_q[4];
//        ceres::QuaternionProduct(relative_q_inv, q_i_j, error_q);

//        residuals[3] = T(2) * error_q[1] / T(q_var);
//        residuals[4] = T(2) * error_q[2] / T(q_var);
//        residuals[5] = T(2) * error_q[3] / T(q_var);

        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3d &iTj_m, double t_var, double q_var);


    Sophus::SE3d iTj_m;
    double t_var;
    double q_var;

};
