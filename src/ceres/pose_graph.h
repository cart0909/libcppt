#pragma once
#include <ceres/autodiff_cost_function.h>
#include <ceres/autodiff_local_parameterization.h>

template<class T>
T NormalizeAngle(const T& angle) {
    if(angle > T(M_PI)) {
        return angle - 2 * M_PI;
    }
    else if(angle < T(-M_PI)) {
        return angle + 2 * M_PI;
    }
    else
        return angle;
}

class AngleLocalParameterization {
public:
    template<class T>
    bool operator()(const T* theta, const T* delta_theta, T* theta_plus_delta) const {
        *theta_plus_delta = NormalizeAngle(*theta + *delta_theta);
        return true;
    }

    static ceres::LocalParameterization* Create() {
        return new ceres::AutoDiffLocalParameterization<AngleLocalParameterization, 1, 1>();
    }
};

// R is raw major
template<class T>
void ypr2R(const T& yaw, const T& pitch, const T& roll, T* R) {
    R[0] = cos(yaw) * cos(pitch);
    R[1] = -sin(yaw) * cos(roll) + cos(yaw) * sin(pitch) * sin(roll);
    R[2] = sin(yaw) * sin(roll) + cos(yaw) * sin(pitch) * cos(roll);
    R[3] = sin(yaw) * cos(pitch);
    R[4] = cos(yaw) * cos(roll) + sin(yaw) * sin(pitch) * sin(roll);
    R[5] = -cos(yaw) * sin(roll) + sin(yaw) * sin(pitch) * cos(roll);
    R[6] = -sin(pitch);
    R[7] = cos(pitch) * sin(roll);
    R[8] = cos(pitch) * cos(roll);
}

template<class T>
void TransposeRotationMatrix(const T* R, T* Rt) {
    Rt[0] = R[0];
    Rt[1] = R[3];
    Rt[2] = R[6];

    Rt[3] = R[1];
    Rt[4] = R[4];
    Rt[5] = R[7];

    Rt[6] = R[2];
    Rt[7] = R[5];
    Rt[8] = R[8];
}

template<class T>
void RotateVector(const T* R, const T* t, T* r_t) {
    r_t[0] = R[0] * t[0] + R[1] * t[1] + R[2] * t[2];
    r_t[1] = R[3] * t[0] + R[4] * t[1] + R[5] * t[2];
    r_t[2] = R[6] * t[0] + R[7] * t[1] + R[8] * t[2];
}

class FourDOFError {
public:
    FourDOFError(double t_x_, double t_y_, double t_z_,
                 double relative_yaw_, double pitch_i_, double roll_i_,
                 double t_cov_, double r_cov_)
        : t_x(t_x_), t_y(t_y_), t_z(t_z_), relative_yaw(relative_yaw_), pitch_i(pitch_i_), roll_i(roll_i_),
          t_cov(t_cov_), r_cov(r_cov_)
    {}

    template<class T>
    bool operator()(const T* const yaw_i, const T* const ti,
                    const T* const yaw_j, const T* const tj,
                    T* residuals) const {
        T tw_ij[3] = {tj[0] - ti[0], tj[1] - ti[1], tj[2] - ti[2]};

        T Rwi[9];
        ypr2R(yaw_i[0], T(pitch_i), T(roll_i), Rwi);

        T Riw[9];
        TransposeRotationMatrix(Rwi, Riw);

        T ti_ij[3];
        RotateVector(Riw, tw_ij, ti_ij);

        residuals[0] = (ti_ij[0] - T(t_x)) / T(t_cov);
        residuals[1] = (ti_ij[1] - T(t_y)) / T(t_cov);
        residuals[2] = (ti_ij[2] - T(t_z)) / T(t_cov);
        residuals[3] = (NormalizeAngle(yaw_j[0] - yaw_i[0] - T(relative_yaw))) / T(r_cov);
        return true;
    }

    static ceres::CostFunction* Create(double t_x, double t_y, double t_z,
                                       double relative_yaw, double pitch_i, double roll_i,
                                       double t_cov = 1, double r_cov = 1) {
        return new ceres::AutoDiffCostFunction<FourDOFError, 4, 1, 3, 1, 3>(
                    new FourDOFError(t_x, t_y, t_z, relative_yaw, pitch_i, roll_i, t_cov, r_cov));
    }

    double t_x, t_y, t_z; // vector i to j observed by i frame
    double relative_yaw;
    double pitch_i, roll_i;
    double t_cov, r_cov;
};

