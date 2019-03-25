#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
// ------------------------ref---------------------
// Plucker Coordinates for Lines in the Space
// Structure-From-Motion Using Lines: Representation, Triangulation and Bundle Adjustment

namespace Plucker {

struct Direction {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Direction() : l(0, 0, 0) {}
    Direction(double a, double b, double c)
        : l(a, b, c) {}
    Direction(const Eigen::Vector3d& l_)
        : l(l_) {}
    Eigen::Vector3d l;
};

struct Point3 {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point3() : p(0, 0, 0) {}
    Point3(double a, double b, double c)
        : p(a, b, c) {}
    Point3(const Eigen::Vector3d& p_)
        : p(p_) {}
    Eigen::Vector3d p;
};

struct Point4 {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Point4() : p(0, 0, 0, 0) {}
    Point4(double a, double b, double c, double d)
        : p(a, b, c, d) {}
    Point4(const Eigen::Vector4d& p_)
        : p(p_) {}
    Eigen::Vector4d p;
};

struct Plane {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Plane() : pi(0, 0, 0, 0) {}
    Plane(double a, double b, double c, double d)
        : pi(a, b, c, d) {}
    Plane(const Eigen::Vector4d& pi_)
        : pi(pi_) {}
    Eigen::Vector4d pi;
};

enum LinesStatus {
    SKEW_LINES,
    PARALLEL_LINES,
    INTERSECT_LINES,
};

class Line {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line() : l(1, 0, 0), m(0, 0, 0) {}
    Line(const Line& line) : l(line.l), m(line.m) {}
    Line(const Eigen::Vector3d& l_, const Eigen::Vector3d& m_) {
        double norm_l = l_.norm();
        l = l_ / norm_l;
        m = m_ / norm_l;
    }
    Line(const Point4& M, const Point4& N) {
        m = M.p.head<3>().cross(N.p.head<3>());
        l = M.p(3) * N.p.head<3>() - N.p(3) * M.p.head<3>();
        double l_norm = l.norm();
        l /= l_norm;
        m /= l_norm;
    }

    Line(const Plane& P, const Plane& Q) {
        m = P.pi.head<3>().cross(Q.pi.head<3>());
        l = Q.pi(3) * P.pi.head<3>() - P.pi(3) * Q.pi.head<3>();
        double l_norm = l.norm();
        l /= l_norm;
        m /= l_norm;
    }

    Line(const Point3& p, const Direction& dir) {
        l = dir.l.normalized();
        m = p.p.cross(l);
    }

    double DistanceFromOrigin() const {
        return m.norm();
    }

    Eigen::Vector3d ClosestPointFromOrigin() const {
        // p_perpendicular
        return l.cross(m);
    }

    Eigen::Vector3d Moment(const Eigen::Vector3d& q) const {
        // mq = m - q x l;
        Eigen::Vector3d mq;
        mq = m - q.cross(l);
        return mq;
    }

    Eigen::Vector3d ClosestPoint(const Eigen::Vector3d& q) const {
        // q_perpendicular
        Eigen::Vector3d mq = Moment(q);
        return q + l.cross(mq);
    }

    // # Corollary 2 #
    // Two lines L1 and L2 are co-planar if and only if the reciprocal product of their
    // Plucker coordinates is zero.
    // (1-8)
    static double ReciprocalProduct(const Line& L1, const Line& L2) {
        // (l1, m1) * (l2, m2) = l1.dot(m2) + l2.dot(m1)
        return L1.l.dot(L2.m) + L2.l.dot(L1.m);
    }

    // # Theorem 1 #
    // (1-10)
    static double Distance(const Line& L1, const Line& L2) {
        Eigen::Vector3d l1xl2 = L1.l.cross(L2.l);
        double norm_l1xl2 =l1xl2.norm();
        double d = 0;
        if(norm_l1xl2 <= std::numeric_limits<double>::min()) {
            d = L1.l.cross(L1.m - L2.m).norm();
        }
        else {
            d = ReciprocalProduct(L1, L2) / norm_l1xl2;
        }
        return d;
    }

    // # Theorem 4 #
    // (1-17), (1-18)
    static bool CommonPerpendicular(const Line& L1, const Line& L2, Line& Lcp, LinesStatus* status = nullptr) {
        double L1_star_L2 = ReciprocalProduct(L1, L2);
        Eigen::Vector3d l1xl2 = L1.l.cross(L2.l);
        double norm_l1xl2 = l1xl2.norm();
        if(L1_star_L2 <= std::numeric_limits<double>::min()) {
            if(norm_l1xl2 <= std::numeric_limits<double>::min()) {
                if(status)
                    *status = PARALLEL_LINES;
                return false;
            }
            else {
                if(status)
                    *status = INTERSECT_LINES;
                Eigen::Vector3d l_cp, m_cp;
                l_cp = l1xl2;
                m_cp = L1.m.cross(L2.l) - L2.m.cross(L1.l);
                Lcp = Line(l_cp, m_cp);
                return true;
            }
        }
        else {
            if(status)
                *status = SKEW_LINES;
            Eigen::Vector3d l_cp, m_cp;
            l_cp = l1xl2;
            m_cp = L1.m.cross(L2.l) - L2.m.cross(L1.l) + ((L1_star_L2 * L1.l.dot(L2.l))/l1xl2.squaredNorm()) * l1xl2;
            return true;
        }
    }

private:
    Eigen::Vector3d l;
    Eigen::Vector3d m;
};
}
