#pragma once
#include <Eigen/Dense>
#include <sophus/se3.hpp>
#include <iostream>
// ------------------------ref---------------------
// Plucker Coordinates for Lines in the Space
// Structure-From-Motion Using Lines: Representation, Triangulation and Bundle Adjustment

namespace Plucker {

enum LineInitMethod {
    POINT_DIR,
    PLUCKER_L_M,
    TWO_POINT,
    TWO_PLANE
};

enum LinesStatus {
    SKEW_LINES,
    PARALLEL_LINES,
    INTERSECT_LINES,
};

class Line {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Line();
    Line(const Line& line);
    Line(const Eigen::Vector3d& a, const Eigen::Vector3d& b, LineInitMethod method);
    Line(const Eigen::Vector4d& a, const Eigen::Vector4d& b, LineInitMethod method);

    void operator=(const Line& line);

    const Eigen::Vector3d& l() const;
    const Eigen::Vector3d& m() const;

    void SetPlucker(const Eigen::Vector3d& l, const Eigen::Vector3d& m);
    double Distance() const;
    double Distance(const Eigen::Vector3d& q) const;

    Eigen::Vector3d ClosestPoint() const;
    Eigen::Vector3d ClosestPoint(const Eigen::Vector3d& q) const;

    friend std::ostream& operator<<(std::ostream& s, const Line& L);
    friend Line operator*(const Sophus::SE3d& T21, const Line& L1);
private:
    Eigen::Vector3d l_;
    Eigen::Vector3d m_;
};

// # Corollary 2 #
// Two lines L1 and L2 are co-planar if and only if the reciprocal product of their
// Plucker coordinates is zero.
// (1-8)
double ReciprocalProduct(const Line& L1, const Line& L2);
// # Theorem 1 #
// (1-10)
double Distance(const Line& L1, const Line& L2);
// # Theorem 4 #
// (1-17), (1-18)
bool CommonPerpendicular(const Line& L1, const Line& L2, Line& Lcp, LinesStatus* status = nullptr);
bool Feet(const Line& L1, const Line& L2, Eigen::Vector3d& p1_star, Eigen::Vector3d& p2_star, LinesStatus* status = nullptr);
}
