#pragma once
#include <ceres/ceres.h>
#include "camera_model/simple_stereo_camera.h"
                                                       //Twcj inv_zi Twci
class ProjectionFactor : public ceres::SizedCostFunction<2, 7, 1, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ProjectionFactor(const SimpleStereoCamPtr& camera,
                     const Eigen::Vector2d& pt_j_,
                     const Eigen::Vector2d& pt_i_);

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector2d pt_j;
    Eigen::Vector3d pt_i_normal_plane;
};
                                                              //Twcj inv_zi Twci
class StereoProjectionFactor : public ceres::SizedCostFunction<3, 7, 1, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StereoProjectionFactor(const SimpleStereoCamPtr& camera,
                           const Eigen::Vector3d& pt_j_,
                           const Eigen::Vector2d& pt_i);

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector3d pt_j; // u v ur
    Eigen::Vector3d pt_i_normal_plane;
};

namespace AutoDiff {

class ProjectionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionFactor(const SimpleStereoCamPtr& camera,
                     const Eigen::Vector2d& pt_j_,
                     const Eigen::Vector2d& pt_i)
        : mpCamera(camera), pt_j(pt_j_) {
        mpCamera->BackProject(pt_i, pt_i_normal_plane);
    }

    template<class T>
    bool operator() (const T* const Twcj_raw, const T* const inv_zi_raw, const T* const Twci_raw, T* residuals_raw) const {
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;

        Eigen::Map<const Sophus::SE3<T>> Twcj(Twcj_raw);
        const T inv_zi = inv_zi_raw[0];
        Eigen::Map<const Sophus::SE3<T>> Twci(Twci_raw);
        Eigen::Map<Vector2T> residuals(residuals_raw);

        Sophus::SE3<T> Tcj_ci = Twcj.inverse() * Twci;
        Vector3T x3Di = pt_i_normal_plane / inv_zi;
        Vector3T x3Dj = Tcj_ci * x3Di;
        Vector2T uv;
        mpCamera->Project2(x3Dj, uv);

        residuals = pt_j - uv;

        return true;
    }

    static ceres::CostFunction* Create(const SimpleStereoCamPtr& camera,
                                       const Eigen::Vector2d& pt_j,
                                       const Eigen::Vector2d& pt_i) {
        return new ceres::AutoDiffCostFunction<ProjectionFactor, 2, 7, 1, 7>(
                    new ProjectionFactor(camera, pt_j, pt_i));
    }

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector2d pt_j;
    Eigen::Vector3d pt_i_normal_plane;
};

class StereoProjectionFactor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StereoProjectionFactor(const SimpleStereoCamPtr& camera,
                           const Eigen::Vector3d& pt_j_,
                           const Eigen::Vector2d& pt_i)
        : mpCamera(camera), pt_j(pt_j_) {
        mpCamera->BackProject(pt_i, pt_i_normal_plane);
    }

    template<class T>
    bool operator()(const T* const Twcj_raw, const T* const inv_zi_raw, const T* const Twci_raw, T* residuals_raw) const {
        using Vector2T = Eigen::Matrix<T, 2, 1>;
        using Vector3T = Eigen::Matrix<T, 3, 1>;

        Eigen::Map<const Sophus::SE3<T>> Twcj(Twcj_raw);
        const T inv_zi = inv_zi_raw[0];
        Eigen::Map<const Sophus::SE3<T>> Twci(Twci_raw);
        Eigen::Map<Vector3T> residuals(residuals_raw);

        Sophus::SE3<T> Tcj_ci = Twcj.inverse() * Twci;
        Vector3T x3Di = pt_i_normal_plane / inv_zi;
        Vector3T x3Dj = Tcj_ci * x3Di;
        Vector3T uv_ur;
        mpCamera->Project3(x3Dj, uv_ur);
        residuals = pt_j - uv_ur;
        return true;
    }

    static ceres::CostFunction* Create(const SimpleStereoCamPtr& camera,
                                       const Eigen::Vector3d& pt_j,
                                       const Eigen::Vector2d& pt_i) {
        return new ceres::AutoDiffCostFunction<StereoProjectionFactor, 3, 7, 1, 7>(
                    new StereoProjectionFactor(camera, pt_j, pt_i));
    }

    SimpleStereoCamPtr mpCamera;
    Eigen::Vector3d pt_j; // u v ur
    Eigen::Vector3d pt_i_normal_plane;
};
}

namespace unary {

class ProjectionFactor : public ceres::SizedCostFunction<2, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    ProjectionFactor(const SimpleStereoCamPtr& camera_, const Eigen::Vector2d& pt_,
                     const Eigen::Vector3d& x3Dw_);

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    SimpleStereoCamPtr camera;
    Eigen::Vector2d pt;
    Eigen::Vector3d x3Dw;
};

class StereoProjectionFactor : public ceres::SizedCostFunction<3, 7>
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    StereoProjectionFactor(const SimpleStereoCamPtr& camera_, const Eigen::Vector3d& pt_,
                           const Eigen::Vector3d& x3Dw_);

    virtual bool Evaluate(double const* const* parameters_raw, double *residual_raw,
                          double **jacobian_raw) const;

    SimpleStereoCamPtr camera;
    Eigen::Vector3d pt;
    Eigen::Vector3d x3Dw;
};

};
