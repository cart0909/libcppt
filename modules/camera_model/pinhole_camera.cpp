#include "pinhole_camera.h"

PinholeCamera::~PinholeCamera() {}

PinholeCamera::PinholeCamera(const std::string& camera_name, int image_width, int image_height,
              double fx_, double fy_, double cx_, double cy_,
              double k1_, double k2_, double p1_, double p2_)
    : CameraBase(PINHOLE, camera_name, 8, image_width, image_height), fx(fx_), fy(fy_), cx(cx_), cy(cy_), k1(k1_), k2(k2_), p1(p1_), p2(p2_)
{
    if((k1 == 0.0) && (k2 == 0.0) && (p1 == 0.0) && (p2 == 0.0)) {
        mbNoDistortion = true;
    }
    else {
        mbNoDistortion = false;
    }

    // Inverse camera intrinstic matrix
    inv_K00 = 1.0 / fx;
    inv_K02 = -cx / fx;
    inv_K11 = 1.0 / fy;
    inv_K12 = -cy / fy;

    K << fx, 0, cx,
            0, fy, cy,
            0, 0, 1;
}

void PinholeCamera::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p) const {
    Eigen::Vector2d p_u, p_d;

    // Project points to the normalised plane
    p_u << P(0) / P(2), P(1) / P(2);

    if(mbNoDistortion) {
        p_d = p_u;
    }
    else {
        // Apply distortion
        Eigen::Vector2d d_u;
        Distortion(p_u, d_u);
        p_d = p_u + d_u;
    }

    // Apply generalised projection matrix
    p << fx * p_d(0) + cx,
         fy * p_d(1) + cy;
}

void PinholeCamera::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p,
                            Eigen::Matrix<double, 2, 3>& J) const {
    Eigen::Vector2d p_u, p_d;

    // Project points to the normalised plane
    double inv_z = 1.0 / P(2);
    double inv_z2 = inv_z * inv_z;
    p_u << P(0) * inv_z, P(1) * inv_z;

    if(mbNoDistortion) {
        p_d = p_u;
        J << fx * inv_z, 0, - fx * P(0) * inv_z2,
             0, fy * inv_z, - fy * P(1) * inv_z2;
    }
    else {
        // Apply distortion
        Eigen::Vector2d d_u;
        Eigen::Matrix2d Jd;
        Eigen::Matrix<double, 2, 3> Jinvz;
        Jinvz << inv_z, 0, - P(0) * inv_z2,
                 0, inv_z, - P(1) * inv_z2;
        Distortion(p_u, d_u, Jd);
        p_d = p_u + d_u;
        J = Jd * Jinvz;
        J.row(0) = fx * J.row(0);
        J.row(1) = fy * J.row(1);
    }

    // Apply generalised projection matrix
    p << fx * p_d(0) + cx,
         fy * p_d(1) + cy;
}

// P(2) = 1
void PinholeCamera::BackProject(const Eigen::Vector2d& p , Eigen::Vector3d& P) const {
    // Lift points to normalised plane
    double mx_d, my_d, mx_u, my_u;

    mx_d = inv_K00 * p(0) + inv_K02;
    my_d = inv_K11 * p(1) + inv_K12;

    if(mbNoDistortion) {
        mx_u = mx_d;
        my_u = my_d;
    }
    else {
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
        my_u = my_d - d_u(0);

        for(int i = 1; i < n; ++i) {
            Distortion(Eigen::Vector2d(mx_u, my_u), d_u);
            mx_u = mx_d - d_u(0);
            my_u = my_d - d_u(1);
        }
    }

    // Obtain a projective ray
    P << mx_u, my_u, 1;
}

// p_d = p_u + d_u
void PinholeCamera::Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u) const {
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);
}

// p_d = p_u + d_u
// J = d(p_d)/d(p_u)
void PinholeCamera::Distortion(const Eigen::Vector2d& p_u, Eigen::Vector2d& d_u,
                               Eigen::Matrix2d& J) const {
    double mx2_u, my2_u, mxy_u, rho2_u, rad_dist_u;

    mx2_u = p_u(0) * p_u(0);
    my2_u = p_u(1) * p_u(1);
    mxy_u = p_u(0) * p_u(1);
    rho2_u = mx2_u + my2_u;
    rad_dist_u = k1 * rho2_u + k2 * rho2_u * rho2_u;
    d_u << p_u(0) * rad_dist_u + 2.0 * p1 * mxy_u + p2 * (rho2_u + 2.0 * mx2_u),
           p_u(1) * rad_dist_u + 2.0 * p2 * mxy_u + p1 * (rho2_u + 2.0 * my2_u);

    double dxdmx = 1.0 + rad_dist_u + k1 * 2.0 * mx2_u + k2 * rho2_u * 4.0 * mx2_u + 2.0 * p1 * p_u(1) + 6.0 * p2 * p_u(0);
    double dydmx = k1 * 2.0 * p_u(0) * p_u(1) + k2 * 4.0 * rho2_u * p_u(0) * p_u(1) + p1 * 2.0 * p_u(0) + 2.0 * p2 * p_u(1);
    double dxdmy = dydmx;
    double dydmy = 1.0 + rad_dist_u + k1 * 2.0 * my2_u + k2 * rho2_u * 4.0 * my2_u + 6.0 * p1 * p_u(1) + 2.0 * p2 * p_u(0);

    J << dxdmx, dxdmy,
         dydmx, dydmy;
}
