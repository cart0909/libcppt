#include <iostream>
#include <Eigen/Dense>
#include "solve_dalm.h"

int main() {
    int n = 10;
    int max_iter = 5000;
    double tol = 0.001;
    double lambda = 0.1;

    double sparsity = 0.1;
    int seed = 100;
    srand(seed);

    Eigen::VectorXd b(4), x(n);
    Eigen::MatrixXd A(4, n);
    b << 0, 0, 1, 1;
    A.setRandom(4, n);
//    A.rightCols(1) << 0, 0, 1, 1;
    A.bottomRows(2) = A.bottomRows(2).array().abs();
    std::cout << A << std::endl;
    SolveDALMFast(A, b, lambda, max_iter, tol, x);
    std::cout << x << std::endl;
    return 0;
}
