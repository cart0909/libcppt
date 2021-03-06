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


IdealPinhole::IdealPinhole(int width, int height, double fx_, double fy_, double cx_, double cy_)
    : Camera(width, height), fx(fx_), fy(fy_), cx(cx_), cy(cy_), inv_fx(1.0f/fx_), inv_fy(1.0f/fy_) {}
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
    P << (p(0) - cx) * inv_fx, (p(1) - cy) * inv_fy, 1;
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

Eigen::Vector2d Pinhole::Distortion(const Eigen::Vector2d& p_u) const {
    Eigen::Vector2d d_u;
    Distortion(p_u, d_u);
    return d_u;
}

Fisheye::Fisheye(int width, int height, double fx, double fy, double cx, double cy,
                 double k1_, double k2_, double k3_, double k4_)
    : IdealPinhole(width, height, fx, fy, cx, cy), k1(k1_), k2(k2_), k3(k3_), k4(k4_) {}
Fisheye::~Fisheye() {}

void Fisheye::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J) const {
    double theta = acos(P(2) / P.norm());
    double phi = atan2(P(1), P(0));
    Eigen::Vector2d p_u = r(theta) * Eigen::Vector2d(cos(phi), sin(phi));
    p << fx * p_u(0) + cx,
         fy * p_u(1) + cy;
}

void Fisheye::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {
    Eigen::Vector2d p_u;
    p_u << (p(0) - cx) * inv_fx,
           (p(1) - cy) * inv_fy;

    const double eps = 1e-10;
    const double p_u_norm = p_u.norm();


    // part 1, derive out phi
    double phi = 0.0;

    if(p_u_norm > eps) {
        phi = atan2(p_u(1), p_u(0));
    }

    // part 2, derive out theta
    Eigen::Matrix9d A = Eigen::Matrix9d::Zero();
    A.block<8, 8>(1, 0).setIdentity();
    A(0, 8) = -p_u_norm;
    A(1, 8) = 1;
    A(3, 8) = k1;
    A(5, 8) = k2;
    A(7, 8) = k3;
    A.col(8) /= k4;

    Eigen::EigenSolver<Eigen::Matrix9d> es(A, false);
    Eigen::VectorXcd eigval = es.eigenvalues();

    bool exist_solution = false;
    double theta_star = std::numeric_limits<double>::max();
    for(int i = 0, n = eigval.rows(); i < n; ++i) {
        if(std::abs(eigval(i).imag()) > eps)
            continue;
        double real = eigval(i).real();

        if(real < -eps) {
            continue;
        }
        else if(real < 0.0f) {
            real = 0.0f;
        }

        if(real < theta_star) {
            theta_star = real;

            if(!exist_solution)
                exist_solution = true;
        }
    }

    double theta = p_u_norm;
    if(exist_solution) {
        theta = theta_star;
    }

    // part 3 restore the x y z without real depth
    P << tan(theta) * cos(phi), // sin(theta) * cos(phi)
         tan(theta) * sin(phi), // sin(theta) * sin(phi)
                             1; // cos(theta)
}

double Fisheye::r(double theta) const {
    double theta_2 = theta * theta;
    double theta_3 = theta * theta_2;
    double theta_5 = theta_2 * theta_3;
    double theta_7 = theta_2 * theta_5;
    double theta_9 = theta_2 * theta_7;

    return theta + k1 * theta_3 + k2 * theta_5 + k3 * theta_7 + k4 * theta_9;
}

IdealOmni::IdealOmni(int width, int height, double fx, double fy, double cx, double cy,
                   double xi_)
    : IdealPinhole(width, height, fx, fy, cx, cy), xi(xi_)
{}

IdealOmni::~IdealOmni() {}

void IdealOmni::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J) const {
    Eigen::Vector3d Ps = P.normalized();
    Eigen::Vector2d p_u = Ps.head(2) / (Ps(2) + xi);
    p << fx * p_u(0) + cx,
         fy * p_u(1) + cy;
}

void IdealOmni::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {
    Eigen::Vector2d p_u((p(0) - cx) * inv_fx, (p(1) - cy) * inv_fy);

    if(xi == 1.0f) {
        P << p_u(0), p_u(1), (1.0 - p_u.squaredNorm()) / 2.0f;
        P /= P(2);
    }
    else {
        double xi_2 = xi * xi;
        double p_u_snorm = p_u.squaredNorm();
        double alpha_s = (xi + std::sqrt(1 + (1 - xi_2) * p_u_snorm)) / (p_u_snorm + 1);
        P << alpha_s * p_u(0),
             alpha_s * p_u(1),
             alpha_s - xi;
        P /= P(2);
    }
}

Omni::Omni(int width, int height, double fx, double fy, double cx, double cy, double xi,
           double k1, double k2, double p1, double p2)
    : IdealPinhole(width, height, fx, fy, cx, cy),
      IdealOmni(width, height, fx, fy, cx, cy, xi),
      Pinhole(width, height, fx, fy, cx, cy, k1, k2, p1, p2)
{}
Omni::~Omni() {}

void Omni::Project(const Eigen::Vector3d& P, Eigen::Vector2d& p, Eigen::Matrix2_3d* J) const {
    Eigen::Vector2d p_u;
    IdealOmni::Project(P, p_u);

    Eigen::Vector2d d_u;
    Distortion(p_u, d_u);

    Eigen::Vector2d p_d = p_u + d_u;

    p << fx * p_d(0) + cx,
         fy * p_d(1) + cy;
}

void Omni::BackProject(const Eigen::Vector2d& p, Eigen::Vector3d& P) const {
    // Lift points to normalised plane
    Eigen::Vector2d m_d, m_u;

    m_d << (p(0) - cx) * inv_fx,
           (p(1) - cy) * inv_fy;

    // Recursive distortion model
    int n = 8;
    Eigen::Vector2d d_u;
    Distortion(m_d, d_u);
    // Approximate value
    m_u = m_d - d_u;

    for(int i = 1; i < n; ++i) {
        Distortion(m_u, d_u);
        m_u = m_d - d_u;
    }

    if(xi == 1.0f) {
        P << m_u(0), m_u(1), (1.0 - m_u.squaredNorm()) / 2.0f;
        P /= P(2);
    }
    else {
        double xi_2 = xi * xi;
        double p_u_snorm = m_u.squaredNorm();
        double alpha_s = (xi + std::sqrt(1 + (1 - xi_2) * p_u_snorm)) / (p_u_snorm + 1);
        P << alpha_s * m_u(0),
             alpha_s * m_u(1),
             alpha_s - xi;
        P /= P(2);
    }
}
