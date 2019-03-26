#include "line.h"
#include <glog/logging.h>

namespace Plucker {
Line::Line() : l_(1, 0, 0), m_(0, 0, 0) {}
Line::Line(const Line& line) : l_(line.l_), m_(line.m_) {}
Line::Line(const Eigen::Vector3d& a, const Eigen::Vector3d& b, LineInitMethod method) {
    Eigen::Vector3d l, m;
    if(method == PLUCKER_L_M) {
        l = a;
        m = b;
    }
    else if (method == POINT_DIR) {
        l = b;
        m = a.cross(b);
    }
    else if(method == TWO_POINT) {
        l = b - a;
        m = a.cross(b);
    }
    else
        throw std::runtime_error("Vector3d constructor only support PLUCKER_L_M, POINT_DIR and TWO_POINT");

    double l_norm = l.norm();
    if(l_norm <= std::numeric_limits<double>::min())
        throw std::runtime_error("l close to zero!!!");
    l_ = l / l_norm;
    m_ = m / l_norm;
}

Line::Line(const Eigen::Vector4d& a, const Eigen::Vector4d& b, LineInitMethod method) {
    Eigen::Vector3d l, m;
    if(method == TWO_POINT) {
        l = a(3) * b.head<3>() - b(3) * a.head<3>();
        m = a.head<3>().cross(b.head<3>());
    }
    else if (method == TWO_PLANE) {
        l = a.head<3>().cross(b.head<3>());
        m = a(3) * b.head<3>() - b(3) * a.head<3>();
    }
    else
        throw std::runtime_error("Vector4d constructor only support TWO_POINT and TWO_PLANE");

    double l_norm = l.norm();
    if(l_norm <= std::numeric_limits<double>::min())
        throw std::runtime_error("l close to zero!!!");
    l_ = l / l_norm;
    m_ = m / l_norm;
}

void Line::operator=(const Line& line) {
    l_ = line.l_;
    m_ = line.m_;
}

const Eigen::Vector3d& Line::l() const {
    return l_;
}

const Eigen::Vector3d& Line::m() const {
    return m_;
}

void Line::SetPlucker(const Eigen::Vector3d& l, const Eigen::Vector3d& m) {
    double l_norm = l.norm();
    if(l_norm <= std::numeric_limits<double>::min())
        throw std::runtime_error("l close to zero!!!");
    l_ = l / l_norm;
    m_ = m / l_norm;
}

double Line::Distance() const {
    return m_.norm();
}

double Line::Distance(const Eigen::Vector3d& q) const {
    Eigen::Vector3d mq = m_ - q.cross(l_);
    return mq.norm();
}

Eigen::Vector3d Line::ClosestPoint() const {
    // p_perpendicular
    return l_.cross(m_);
}

Eigen::Vector3d Line::ClosestPoint(const Eigen::Vector3d& q) const {
    // q_perpendicular
    Eigen::Vector3d mq = m_ - q.cross(l_);
    return q + l_.cross(mq);
}

std::ostream& operator<<(std::ostream& s, Line& L) {
    s << "l(" << L.l()(0) << "," << L.l()(1) << "," << L.l()(2) <<
       ") m(" << L.m()(0) << "," << L.m()(1) << "," << L.m()(2) << ").";
    return s;
}

// # Corollary 2 #
// Two lines L1 and L2 are co-planar if and only if the reciprocal product of their
// Plucker coordinates is zero.
// (1-8)
double ReciprocalProduct(const Line& L1, const Line& L2) {
    // (l1, m1) * (l2, m2) = l1.dot(m2) + l2.dot(m1)
    return L1.l().dot(L2.m()) + L2.l().dot(L1.m());
}

// # Theorem 1 #
// (1-10)
double Distance(const Line& L1, const Line& L2) {
    const Eigen::Vector3d &l1 = L1.l(), &l2 = L2.l(), &m1 = L1.m(), &m2 = L2.m();
    Eigen::Vector3d l1xl2 = l1.cross(l2);
    double norm_l1xl2 =l1xl2.norm();
    double d = 0;
    if(norm_l1xl2 <= std::numeric_limits<double>::min()) {
        double s = l1.dot(l2);
        d = l1.cross(m1 - m2 / s).norm();
    }
    else {
        d = std::abs(ReciprocalProduct(L1, L2)) / norm_l1xl2;
    }
    return d;
}

// # Theorem 4 #
// (1-17), (1-18)
bool CommonPerpendicular(const Line& L1, const Line& L2, Line& Lcp, LinesStatus* status) {
    const Eigen::Vector3d &l1 = L1.l(), &l2 = L2.l(), &m1 = L1.m(), &m2 = L2.m();
    double L1_star_L2 = ReciprocalProduct(L1, L2);
    Eigen::Vector3d l1xl2 = l1.cross(l2);
    double norm_l1xl2 = l1xl2.norm();
    if(std::abs(L1_star_L2) <= std::numeric_limits<double>::min()) {
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
            m_cp = m1.cross(l2) - m2.cross(l1);
            Lcp.SetPlucker(l_cp, m_cp);
            return true;
        }
    }
    else {
        if(status)
            *status = SKEW_LINES;
        Eigen::Vector3d l_cp, m_cp;
        l_cp = l1xl2;
        m_cp = m1.cross(l2) - m2.cross(l1) + ((L1_star_L2 * l1.dot(l2))/(norm_l1xl2 * norm_l1xl2)) * l1xl2;
        Lcp.SetPlucker(l_cp, m_cp);
        return true;
    }
}

bool Feet(const Line& L1, const Line& L2, Eigen::Vector3d& p1_star, Eigen::Vector3d& p2_star, LinesStatus* status) {
    const Eigen::Vector3d &l1 = L1.l(), &l2 = L2.l(), &m1 = L1.m(), &m2 = L2.m();
    double L1_star_L2 = ReciprocalProduct(L1, L2);
    Eigen::Vector3d l1xl2 = l1.cross(l2);
    double norm_l1xl2 = l1xl2.norm();

    if(std::abs(L1_star_L2) <= std::numeric_limits<double>::min()) {
        if(norm_l1xl2 <= std::numeric_limits<double>::min()) {
            if(status)
                *status = PARALLEL_LINES;
            return false;
        }
        else {
            if(status)
                *status = INTERSECT_LINES;
            Eigen::Matrix3d I3 = Eigen::Matrix3d::Identity();
            p1_star = ((m1.dot(l2)*I3 + l1*m2.transpose() - l2*m1.transpose()) * l1xl2)/(norm_l1xl2 * norm_l1xl2);
            p2_star = p1_star;
            return true;
        }
    }
    else {
        if(status)
            *status = SKEW_LINES;
        double norm_l1xl2_2 = norm_l1xl2 * norm_l1xl2;
        p1_star = (-m1.cross(l2.cross(l1xl2)) + (m2.dot(l1xl2))*l1) / norm_l1xl2_2;
        p2_star = ( m2.cross(l1.cross(l1xl2)) - (m1.dot(l1xl2))*l2) / norm_l1xl2_2;
        return true;
    }
}

}
