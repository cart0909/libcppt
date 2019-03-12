#pragma once
#include <Eigen/Dense>
// reference : https://people.eecs.berkeley.edu/~yang/software/l1benchmark/
void SolveDALMFast(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, double lambda,
                   int max_iter, double tol, Eigen::VectorXd& x);
