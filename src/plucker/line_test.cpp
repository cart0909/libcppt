#include <iostream>
#include <glog/logging.h>
#include "plucker/line.h"
using namespace Plucker;

int main() {
    Eigen::Vector3d A(1, 0, 0), B(1, 1, 1), C(0, 1, 3), D(-1, 2, 1);
    Line3d l1(A, B, POINT_DIR), l2(C, D, POINT_DIR);
    Eigen::Map<Line3d> l3(l1.data());
    Sophus::SE3d T21;
    T21 = Sophus::SE3d::rotX(M_PI/6) * Sophus::SE3d::rotY(M_PI/2) * Sophus::SE3d::rotX(M_PI/5);
    T21.translation() << 6, 8, -7;
    std::cout << l1 << std::endl;
    std::cout << T21 * l1 << std::endl;

    Eigen::Vector4d delta = Eigen::Vector4d::Random();
    std::cout << l1 * delta << std::endl;
    std::cout << l3 * delta << std::endl;
    return 0;
}
