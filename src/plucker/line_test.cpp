#include <iostream>
#include <glog/logging.h>
#include "plucker/line.h"
using namespace Plucker;

int main() {
    Eigen::Vector3d A(1, 0, 0), B(1, 1, 1), C(0, 1, 3), D(-1, 2, 1);
    Line3d l1(A, B, POINT_DIR), l2(C, D, POINT_DIR);
    Eigen::Map<Line3d> l3(l2.data());
//    Sophus::SE3d T21;
//    T21 = Sophus::SE3d::rotX(M_PI/6) * Sophus::SE3d::rotY(M_PI/2) * Sophus::SE3d::rotX(M_PI/5);
//    T21.translation() << 6, 8, -7;
    std::cout << ReciprocalProduct(l1, l3) << std::endl;
    std::cout << Distance(l1, l3) << std::endl;
    Line3d l4;
    CommonPerpendicular(l1, l2, l4);
    Eigen::Vector3d p1, p2;
    Feet(l1, l2, p1, p2);
    std::cout << p1.transpose() - p2.transpose()<< std::endl;
    std::cout << l4.l().transpose().cross(p1.transpose() - p2.transpose()).norm() << std::endl;
    return 0;
}
