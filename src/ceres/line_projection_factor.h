#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <ceres/sized_cost_function.h>
#include <ceres/autodiff_cost_function.h>
#include "util.h"
#include "plucker/line.h"

namespace Plucker {

class LineProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 4>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    bool Evaluate(double const* const* parameters_raw, double* residuals_raw, double** jacobians_raw) const {
        Eigen::Matrix2d sqrt_info;
        Eigen::Vector3d spt, ept;
        // parameters [0]: Twb
        //            [2]: Tbc
        //            [3]: Lw
        Eigen::Map<const Sophus::SE3d> Twb(parameters_raw[0]), Tbc(parameters_raw[1]);
        Eigen::Map<const Plucker::Line3d> Lw(parameters_raw[2]);
        Eigen::Map<Eigen::Vector2d> residuals(residuals_raw);

        Sophus::SE3d Tcw = (Twb * Tbc).inverse();
        Plucker::Line3d Lc = Tcw * Lw;
        Eigen::Vector3d l = Lc.m() / Lc.m().head<2>().norm(); // 2d line in normal plane equal to Lc normal vector
        residuals << l.dot(spt),
                     l.dot(ept);

        residuals = sqrt_info * residuals;

        if(jacobians_raw) {
            Eigen::Matrix2_3d dr_dmc;
            Eigen::Vector3d mc = Lc.m();
            double m01 = mc.head<2>().norm();
            double inv_m01 = 1.0f / m01, inv_m01_2 = inv_m01 * inv_m01;
            dr_dmc << spt(0)*inv_m01-mc(0)*residuals(0)*inv_m01_2, spt(1)*inv_m01-mc(1)*residuals(0)*inv_m01_2, inv_m01,
                      ept(0)*inv_m01-mc(0)*residuals(1)*inv_m01_2, ept(1)*inv_m01-mc(1)*residuals(1)*inv_m01_2, inv_m01;
            dr_dmc = sqrt_info * dr_dmc;

            if(jacobians_raw[0]) {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jwb(jacobians_raw[0]);
                Eigen::Matrix<double, 2, 6> J;
                Jwb.rightCols<1>().setZero();
            }

            if(jacobians_raw[1]) {
                Eigen::Map<Eigen::Matrix<double, 2, 7, Eigen::RowMajor>> Jbc(jacobians_raw[1]);
                Eigen::Matrix<double, 2, 6> J;
                Jbc.rightCols<1>().setZero();
            }

            if(jacobians_raw[2]) {
                Eigen::Map<Eigen::Matrix<double, 2, 6, Eigen::RowMajor>> Jline(jacobians_raw[2]);
                Jline.rightCols<1>().setZero();
            }
        }

        return true;
    }
};

}

class LineProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 2>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineProjectionFactor(const Eigen::Vector3d& spi_, const Eigen::Vector3d& epi_,
                         const Eigen::Vector3d& spj_, const Eigen::Vector3d& epj_,
                         double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d spi, epi;
    Eigen::Vector3d spj, epj;
    Eigen::Matrix2d sqrt_info;
};

class LineSlaveProjectionFactor : public ceres::SizedCostFunction<2, 7, 7, 7, 7, 2>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSlaveProjectionFactor(const Eigen::Vector3d& sp_mi_, const Eigen::Vector3d& ep_mi_,
                              const Eigen::Vector3d& sp_sj_, const Eigen::Vector3d& ep_sj_,
                              double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d sp_mi, ep_mi;
    Eigen::Vector3d sp_sj, ep_sj;
    Eigen::Matrix2d sqrt_info;
};

class LineSelfProjectionFactor : public ceres::SizedCostFunction<2, 7, 2>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSelfProjectionFactor(const Eigen::Vector3d& sp_l_, const Eigen::Vector3d& ep_l_,
                             const Eigen::Vector3d& sp_r_, const Eigen::Vector3d& ep_r_,
                             double focal_length);

    bool Evaluate(double const * const* parameters_raw,
                  double* residuals_raw,
                  double** jacobians_raw) const;

    Eigen::Vector3d spl, epl;
    Eigen::Vector3d spr, epr;
    Eigen::Matrix2d sqrt_info;
};

namespace autodiff {

class LineProjectionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineProjectionFactor(const Eigen::Vector3d& spi_, const Eigen::Vector3d& epi_,
                         const Eigen::Vector3d& spj_, const Eigen::Vector3d& epj_,
                         double focal_length)
        : spi(spi_), epi(epi_), spj(spj_), epj(epj_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
    }

    template<class T>
    bool operator()(const T* const Twbi_raw, const T* const Twbj_raw, const T* const Tbc_raw,
                    const T* const line_raw, T* residuals_raw) const {
        Eigen::Map<const Sophus::SE3<T>> Twbi(Twbi_raw);
        Eigen::Map<const Sophus::SE3<T>> Twbj(Twbj_raw);
        Eigen::Map<const Sophus::SE3<T>> Tbc(Tbc_raw);
        T P_inv_zi = line_raw[0], Q_inv_zi = line_raw[1];
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Sophus::SE3<T> Tcb = Tbc.inverse(), Tbjw = Twbj.inverse();
        Sophus::SE3<T> Tcjci = Tcb * Tbjw * Twbi * Tbc;
        Eigen::Matrix<T, 3, 1> Pcj = Tcjci * (spi.cast<T>() / P_inv_zi),
                               Qcj = Tcjci * (epi.cast<T>() / Q_inv_zi);
        T P_inv_zj = T(1.0f) / Pcj(2), Q_inv_zj = T(1.0f) / Qcj(2);

        Eigen::Matrix<T, 3, 1> l = (spj.cross(epj)).cast<T>();
        l /= l.template head<2>().norm();

        residuals << l.dot(Pcj * P_inv_zj),
                     l.dot(Qcj * Q_inv_zj);

        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& spi, const Eigen::Vector3d& epi,
                                       const Eigen::Vector3d& spj, const Eigen::Vector3d& epj,
                                       double focal_length) {
        return new ceres::AutoDiffCostFunction<LineProjectionFactor, 2, 7, 7, 7, 2>(
                    new LineProjectionFactor(spi, epi, spj, epj, focal_length));
    }

    Eigen::Vector3d spi, epi;
    Eigen::Vector3d spj, epj;
    Eigen::Matrix2d sqrt_info;
};

class LineSlaveProjectionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSlaveProjectionFactor(const Eigen::Vector3d& sp_mi_, const Eigen::Vector3d& ep_mi_,
                              const Eigen::Vector3d& sp_sj_, const Eigen::Vector3d& ep_sj_,
                              double focal_length)
        : sp_mi(sp_mi_), ep_mi(ep_mi_), sp_sj(sp_sj_), ep_sj(ep_sj_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
    }

    template<class T>
    bool operator()(const T* const Twbi_raw, const T* const Twbj_raw, const T* const Tbc_raw,
                    const T* const Tsl_raw, const T* const line_raw, T* residuals_raw) const {
        Eigen::Map<const Sophus::SE3<T>> Twbi(Twbi_raw);
        Eigen::Map<const Sophus::SE3<T>> Twbj(Twbj_raw);
        Eigen::Map<const Sophus::SE3<T>> Tbc(Tbc_raw);
        Eigen::Map<const Sophus::SE3<T>> Tsl(Tsl_raw);
        T P_inv_zi = line_raw[0], Q_inv_zi = line_raw[1];
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Sophus::SE3<T> Tcb = Tbc.inverse(), Tbjw = Twbj.inverse();
        Sophus::SE3<T> Tsjci = Tsl * Tcb * Tbjw * Twbi * Tbc;
        Eigen::Matrix<T, 3, 1> Psj = Tsjci * (sp_mi.cast<T>() / P_inv_zi),
                               Qsj = Tsjci * (ep_mi.cast<T>() / Q_inv_zi);
        T P_inv_zj = T(1.0f) / Psj(2), Q_inv_zj = T(1.0f) / Qsj(2);

        Eigen::Matrix<T, 3, 1> l = (sp_sj.cross(ep_sj)).cast<T>();
        l /= l.template head<2>().norm();

        residuals << l.dot(Psj * P_inv_zj),
                     l.dot(Qsj * Q_inv_zj);

        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& sp_mi, const Eigen::Vector3d& ep_mi,
                                       const Eigen::Vector3d& sp_sj, const Eigen::Vector3d& ep_sj,
                                       double focal_length) {
        return new ceres::AutoDiffCostFunction<LineSlaveProjectionFactor, 2, 7, 7, 7, 7, 2>(
                    new LineSlaveProjectionFactor(sp_mi, ep_mi, sp_sj, ep_sj, focal_length));
    }

    Eigen::Vector3d sp_mi, ep_mi;
    Eigen::Vector3d sp_sj, ep_sj;
    Eigen::Matrix2d sqrt_info;
};

class LineSelfProjectionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LineSelfProjectionFactor(const Eigen::Vector3d& spi_, const Eigen::Vector3d& epi_,
                             const Eigen::Vector3d& spj_, const Eigen::Vector3d& epj_,
                             double focal_length)
        : spi(spi_), epi(epi_), spj(spj_), epj(epj_)
    {
        sqrt_info = Eigen::Matrix2d::Identity() * focal_length / 1.5;
    }

    template<class T>
    bool operator()(const T* const Trl_raw, const T* const line_raw, T* residuals_raw) const {
        Eigen::Map<const Sophus::SE3<T>> Trl(Trl_raw);
        T P_inv_zi = line_raw[0], Q_inv_zi = line_raw[1];
        Eigen::Map<Eigen::Matrix<T, 2, 1>> residuals(residuals_raw);

        Eigen::Matrix<T, 3, 1> Pcj = Trl * (spi.cast<T>() / P_inv_zi),
                               Qcj = Trl * (epi.cast<T>() / Q_inv_zi);
        T P_inv_zj = T(1.0f) / Pcj(2), Q_inv_zj = T(1.0f) / Qcj(2);

        Eigen::Matrix<T, 3, 1> l = (spj.cross(epj)).cast<T>();
        l /= l.template head<2>().norm();

        residuals << l.dot(Pcj * P_inv_zj),
                     l.dot(Qcj * Q_inv_zj);

        residuals = sqrt_info.cast<T>() * residuals;
        return true;
    }

    static ceres::CostFunction* Create(const Eigen::Vector3d& spi, const Eigen::Vector3d& epi,
                                       const Eigen::Vector3d& spj, const Eigen::Vector3d& epj,
                                       double focal_length) {
        return new ceres::AutoDiffCostFunction<LineSelfProjectionFactor, 2, 7, 2>(
                    new LineSelfProjectionFactor(spi, epi, spj, epj, focal_length));
    }

    Eigen::Vector3d spi, epi;
    Eigen::Vector3d spj, epj;
    Eigen::Matrix2d sqrt_info;
};
}
