#include <iostream>
#include <glog/logging.h>
#include "plucker/line.h"
using namespace Plucker;

int main() {
    Eigen::Vector3d p1(3, -4, -2), d1(1, -2, 1), p2(-9, 2, 0), d2(4, -1, 2);
    Line3d l1(p1, d1, POINT_DIR), l2(p2, d2, POINT_DIR);
    std::cout << Distance(l1, l2) << std::endl;

    p1 << 1, 2, -4;
    d1 << 3, -1, 1;
    p2 << -4, 6, -7;
    l1 = Line3d(p1, d1, POINT_DIR);
    std::cout << l1.Distance(p2) << std::endl;
    std::cout << l1.ClosestPoint(p2).transpose() << std::endl;

    p1 << 2, 0, 5;
    d1 << -1, 2, -2;
    p2 << -1, 1, 2;
    d2 << 2, 1, 1;
    l1 = Line3d(p1, d1, POINT_DIR);
    l2 = Line3d(p2, d2, POINT_DIR);
    Eigen::Vector3d p1_star, p2_star;
    LinesStatus status;
    Feet(l1, l2, p1_star, p2_star, &status);
    std::cout << p1_star.transpose() << std::endl;
    std::cout << p2_star.transpose() << std::endl;
    std::cout << status << std::endl;

    p1 << 3, 1, -1;
    d1 << 1, 2, 1;
    p2 << 2, 0, 1;
    d2 << 1, 1, 1;
    l1 = Line3d(p1, d1, POINT_DIR);
    l2 = Line3d(p2, d2, POINT_DIR);
    Feet(l1, l2, p1_star, p2_star, &status);
    std::cout << p1_star.transpose() << std::endl;
    std::cout << p2_star.transpose() << std::endl;
    std::cout << status << std::endl;


    p1 << 3, 0, -2;
    d1 << 1, -1, -2;
    p2 << 9, -2 , -1;
    d2 << 2, -2 ,-4;
    l1 = Line3d(p1, d1, POINT_DIR);
    l2 = Line3d(p2, d2, POINT_DIR);
    std::cout << pow(Distance(l1, l2), 2) << std::endl;

    p1 << 5, -7, 1;
    d1 << 3, -6, -2;
    p2 << 1, 0, -5;
    d2 << 3, 2, 2;
    l1 = Line3d(p1, d1, POINT_DIR);
    l2 = Line3d(p2, d2, POINT_DIR);
    std::cout << pow(Distance(l1, l2), 2) << std::endl;

    return 0;
}
