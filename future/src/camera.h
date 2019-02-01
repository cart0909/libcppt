#pragma once
#include "util.h"

class Camera {
public:
    Camera(int width_, int height_);
    virtual ~Camera();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J = nullptr) const = 0;
    Eigen::Vector2d Project(const Eigen::Vector3d& P) const;

    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const = 0;
    Eigen::Vector3d BackProject(const Eigen::Vector2d& p) const;

    // pd_u = p_u + d_u
    virtual double f() const = 0;

    const int width;
    const int height;
};

SMART_PTR(Camera)

class IdealPinhole : public Camera {
public:
    IdealPinhole(int width, int height, double fx_, double fy_, double cx_, double cy_);
    virtual ~IdealPinhole();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J = nullptr) const;
    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    virtual double f() const;

    const double fx;
    const double fy;
    const double cx;
    const double cy;
    const double inv_fx;
    const double inv_fy;
};

SMART_PTR(IdealPinhole)

class Pinhole : public IdealPinhole {
public:
    Pinhole(int width, int height, double fx, double fy, double cx, double cy,
            double k1_, double k2_, double p1_, double p2_);
    virtual ~Pinhole();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J = nullptr) const;
    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;
    // pd_u = p_u + d_u
    void Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J = nullptr) const;
    Eigen::Vector2d Distortion(const Eigen::Vector2d& p_u) const;

    const double k1;
    const double k2;
    const double p1;
    const double p2;
};

SMART_PTR(Pinhole)

class Fisheye : public IdealPinhole {
public:
    Fisheye(int width, int height, double fx, double fy, double cx, double cy,
            double k1_, double k2_, double k3_, double k4_);
    virtual ~Fisheye();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J = nullptr) const;
    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;

    double r(double theta) const;

    const double k1;
    const double k2;
    const double k3;
    const double k4;
};

SMART_PTR(Fisheye);

class IdelOmni : public IdealPinhole {
public:
    IdelOmni(int width, int height, double fx, double fy, double cx, double cy,
             double xi_);
    virtual ~IdelOmni();

    virtual void Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J = nullptr) const;
    virtual void BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const;

    const double xi;
};

SMART_PTR(IdelOmni)
