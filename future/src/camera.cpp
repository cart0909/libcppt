#include "camera.h"

Camera::Camera(int width_, int height_) : width(width_), height(height_) {}
Camera::~Camera() {}

Eigen::Vector2d Camera::Project(const Eigen::Vector3d& P) const {
    Eigen::Vector2d p;
    Project(P, p);
    return p;
}

Eigen::Vector3d Camera::BackProject(const Eigen::Vector2d& p) const {
    Eigen::Vector3d P;
    BackProject(p, P);
    return P;
}

Eigen::Vector2d Camera::Distortion(const Eigen::Vector2d& p_u) const {
    Eigen::Vector2d d_u;
    Distortion(p_u, d_u);
    return d_u;
}

IdealPinhole::IdealPinhole(int width, int height, double fx_, double fy_, double cx_, double cy_)
    : Camera(width, height), fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}
IdealPinhole::~IdealPinhole() {}

void IdealPinhole::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J) const {
    const double inv_z = 1.0f / P(2);
    p << fx * P(0) * inv_z + cx,
         fy * P(1) * inv_z + cy;

    if(J) {
        const double inv_z2 = inv_z * inv_z;
        *J << fx * inv_z, 0, -fx * P(0) * inv_z2,
              0, fy * inv_z, -fy * P(1) * inv_z2;
    }
}

void IdealPinhole::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {
    const double inv_fx = 1.0f / fx;
    const double inv_fy = 1.0f / fy;
    P << (p(0) - cx) * inv_fx, (p(1) - cy) * inv_fy, 1;
}

// p_d = p_u + d_u
// J = d(p_d)/d(p_u)
void IdealPinhole::Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J) const {
    d_u.setZero();
    if(J) {
        J->setIdentity();
    }
}

double IdealPinhole::f() const {
    return fx;
}

Pinhole::Pinhole(int width, int height, double fx, double fy, double cx, double cy,
                 double k1_, double k2_, double p1_, double p2_)
    : IdealPinhole(width, height, fx, fy, cx, cy), k1(k1_), k2(k2_), p1(p1_), p2(p2_) {}
Pinhole::~Pinhole() {}

void Pinhole::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J) const {
    Eigen::Vector2d p_u, p_d;

    if(J) {
        // Project points to the normalised plane
        double inv_z = 1.0 / P(2);
        double inv_z2 = inv_z * inv_z;
        p_u << P(0) * inv_z, P(1) * inv_z;

        Eigen::Vector2d d_u;
        Eigen::Matrix2d Jd;
        Eigen::Matrix2_3d Jinvz;
        Jinvz << inv_z, 0, - P(0) * inv_z2,
                 0, inv_z, - P(1) * inv_z2;
        Distortion(p_u, d_u, &Jd);
        p_d = p_u + d_u;
        *J = Jd * Jinvz;
        J->row(0) = fx * J->row(0);
        J->row(1) = fy * J->row(1);

        // Apply generalised projection matrix
        p << fx * p_d(0) + cx,
             fy * p_d(1) + cy;
    }
    else {
        // Project points to the normalised plane
        double inv_z = 1.0 / P(2);
        p_u << P(0) * inv_z, P(1) * inv_z;

        // Apply distortion
        Eigen::Vector2d d_u;
        Distortion(p_u, d_u);
        p_d = p_u + d_u;

        // Apply generalised projection matrix
        p << fx * p_d(0) + cx,
             fy * p_d(1) + cy;
    }
}

void Pinhole::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {
    // Lift points to normalised plane
    double mx_d, my_d, mx_u, my_u;

    const double inv_fx = 1.0f / fx;
    const double inv_fy = 1.0f / fy;
    mx_d = (p(0) - cx) * inv_fx;
    my_d = (p(1) - cy) * inv_fy;

#if 0
    // Apply inverse distortion model
    // proposed by Heikkila
    mx2_d = mx_d*mx_d;
    my2_d = my_d*my_d;
    mxy_d = mx_d*my_d;
    rho2_d = mx2_d+my2_d;
    rho4_d = rho2_d*rho2_d;
    radDist_d = k1*rho2_d+k2*rho4_d;
    Dx_d = mx_d*radDist_d + p2*(rho2_d+2*mx2_d) + 2*p1*mxy_d;
    Dy_d = my_d*radDist_d + p1*(rho2_d+2*my2_d) + 2*p2*mxy_d;
    inv_denom_d = 1/(1+4*k1*rho2_d+6*k2*rho4_d+8*p1*my_d+8*p2*mx_d);

    mx_u = mx_d - inv_denom_d*Dx_d;
    my_u = my_d - inv_denom_d*Dy_d;
#endif
    // Recursive distortion model
    int n = 8;
    Eigen::Vector2d d_u;
    Distortion(Eigen::Vector2d(mx_d, my_d), d_u);
    // Approximate value
    mx_u = mx_d - d_u(0);
    my_u = my_d - d_u(1);

    for(int i = 1; i < n; ++i) {
        Distortion(Eigen::Vector2d(mx_u, my_u), d_u);
        mx_u = mx_d - d_u(0);
        my_u = my_d - d_u(1);
    }

    // Obtain a projective ray
    P << mx_u, my_u, 1;
}

// p_d = p_u + d_u
// J = d(p_d)/d(p_u)
void Pinhole::Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J) const {
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    if(J) {
        double dxdmx = 1.0 + rad_dist_u + k1 * 2.0 * mx2_u + k2 * rho2_u * 4.0 * mx2_u + 2.0 * p1 * p_u(1) + 6.0 * p2 * p_u(0);
        double dydmx = k1 * 2.0 * p_u(0) * p_u(1) + k2 * 4.0 * rho2_u * p_u(0) * p_u(1) + p1 * 2.0 * p_u(0) + 2.0 * p2 * p_u(1);
        double dxdmy = dydmx;
        double dydmy = 1.0 + rad_dist_u + k1 * 2.0 * my2_u + k2 * rho2_u * 4.0 * my2_u + 6.0 * p1 * p_u(1) + 2.0 * p2 * p_u(0);

        *J << dxdmx, dxdmy,
              dydmx, dydmy;
    }
}

Fisheye::Fisheye(int width, int height, double fx, double fy, double cx, double cy,
                 double k1_, double k2_, double k3_, double k4_)
    : IdealPinhole(width, height, fx, fy, cx, cy), k1(k1_), k2(k2_), k3(k3_), k4(k4_) {}
Fisheye::~Fisheye() {}

void Fisheye::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J) const {

}

void Fisheye::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {

}

// pd_u = p_u + d_u
void Fisheye::Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u, Eigen::Matrix2d* J) const {

}

