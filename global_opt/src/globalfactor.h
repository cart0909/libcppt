#pragma once
#include <ceres/ceres.h>
#include "ceres/local_parameterization_se3.h"
#include "ceres/local_parameterization_so3.h"
#include "util.h"
#include <ceres/autodiff_cost_function.h>

class TError
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    TError(const double t_x, const double t_y, const double t_z, const double var);


    template <typename T>
    bool operator()(const T* wvio_t_i_, T* residuals) const
    {
        Eigen::Map<const Sophus::Vector3<T>> wvio_t_i(wvio_t_i_);
        residuals[0] = (wvio_t_i.x() - T(t_x)) / T(var);
        residuals[1] = (wvio_t_i.y() - T(t_y)) / T(var);
        residuals[2] = (wvio_t_i.z() - T(t_z)) / T(var);
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

    bool operator()(const T*  wvioPi_, const T*  wvioQi_, const T* wvioPj_, const T* wvioQj_, T* residuals) const
    {
        Eigen::Map<const Sophus::SO3<T>> wvioQi(wvioQi_);
        Eigen::Map<const Sophus::SO3<T>> wvioQj(wvioQj_);
        Sophus::Vector3<T> t_w_ij;
        t_w_ij.x() = wvioPj_[0] - wvioPi_[0];
        t_w_ij.y() = wvioPj_[1] - wvioPi_[1];
        t_w_ij.z() = wvioPj_[2] - wvioPi_[2];

        Sophus::SO3<T> i_Q_wvio = wvioQi.inverse();
        Sophus::Vector3<T> t_i_ij;
        t_i_ij = i_Q_wvio * t_w_ij;

        residuals[0] = (t_i_ij.x() - iTj_m.translation().x()) / T(t_var);
        residuals[1] = (t_i_ij.y() - iTj_m.translation().y()) / T(t_var);
        residuals[2] = (t_i_ij.z() - iTj_m.translation().z()) / T(t_var);

        Sophus::SO3<T> q_error = iTj_m.so3().inverse() * (i_Q_wvio * wvioQj);

        residuals[3] = T(2) * q_error.unit_quaternion().x() / T(q_var);
        residuals[4] = T(2) * q_error.unit_quaternion().y() / T(q_var);
        residuals[5] = T(2) * q_error.unit_quaternion().z() / T(q_var);
        return true;
    }

    static ceres::CostFunction* Create(const Sophus::SE3d &iTj_m, double t_var, double q_var);


    Sophus::SE3d iTj_m;
    double t_var;
    double q_var;

};
