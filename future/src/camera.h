#pragma once
#include "util.h"

class Camera {
public:
    Camera(uint width_, uint height_);
    virtual ~Camera();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix23d* J = nullptr) const = 0;
    virtual Eigen::Vector2d Project(const Eigen::Vector3d& P) const = 0;

    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const = 0;
    virtual Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const = 0;

    // pd_u = p_u + d_u
    virtual void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J = nullptr) const = 0;
    virtual Eigen::Vector2d Distortion(const Eigen::Vector2d& p_u) const = 0;
    virtual double f() const = 0;

    const uint width;
    const uint height;
};

class IdealPinhole : public Camera {
public:
    IdealPinhole(uint width, uint height, double fx_, double fy_, double cx_, double cy_);
    virtual ~IdealPinhole();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix23d* J = nullptr) const;
    virtual Eigen::Vector2d Project(const Eigen::Vector3d& P) const;

    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    virtual Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const;

    // pd_u = p_u + d_u
    virtual void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J = nullptr) const;
    virtual Eigen::Vector2d Distortion(const Eigen::Vector2d& p_u) const;

    virtual double f() const;

    const double fx;
    const double fy;
    const double cx;
    const double cy;
};

class Pinhole : public IdealPinhole {
public:
    Pinhole(uint width, uint height, double fx, double fy, double cx, double cy,
            double k1_, double k2_, double p1_, double p2_);
    virtual ~Pinhole();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix23d* J = nullptr) const;
    virtual Eigen::Vector2d Project(const Eigen::Vector3d& P) const;

    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    virtual Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const;

    // pd_u = p_u + d_u
    virtual void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J = nullptr) const;
    virtual Eigen::Vector2d Distortion(const Eigen::Vector2d& p_u) const;

    const double k1;
    const double k2;
    const double p1;
    const double p2;
};
