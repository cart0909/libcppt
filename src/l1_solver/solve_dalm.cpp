#include "solve_dalm.h"

void SolveDALMFast(const Eigen::MatrixXd& A, const Eigen::VectorXd& b, double lambda,
                   int max_iter, double tol, Eigen::VectorXd& x) {
    // A: m x n
    // b: m x 1
    int m = A.rows(), n = A.cols();
    if(m != b.size())
        throw std::runtime_error("size of A and b not align");

    x = Eigen::VectorXd::Zero(n); // n x 1

    // beta = l1norm(b) / m
    double beta = b.lpNorm<1>() / m;
    double beta_inv = 1.0f / beta;

    int n_iter = 0;
    Eigen::VectorXd y = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd z = Eigen::VectorXd::Zero(n);

    bool converged_main = false;

    // At_y = A' * y;
    Eigen::VectorXd At_y = A.transpose() * y; // n x 1

    do {
        ++n_iter;

        Eigen::VectorXd x_old = x; // n x 1

        // % update z
        // temp = At_y + x * beta_inv
        Eigen::VectorXd temp = At_y + x * beta_inv; // n x 1
        // z = sign(temp) .* min(1, abs(temp))
        z = temp.array().sign().array() * temp.array().abs().min(1).array(); // n x 1

        // % compute At * y
        // g = lambda * y - b + A * (beta * (temp - z) + x)
        Eigen::VectorXd g = lambda * y - b + A * (beta * (temp - z) + x); // m x 1
        Eigen::VectorXd Ag = A.transpose() * g; // n x 1
        // alpha = (g'*g)/(lambda * g'*g + beta * Ag'*Ag)
        double g_sn = g.squaredNorm();
        double alpha = g_sn / (lambda * g_sn + beta * Ag.squaredNorm());
        // y = y - alpha * g
        y = y - alpha * g; // m x 1

        // At_y = A' * y;
        At_y = A.transpose() * y; // n x 1

        // % update x
        // x = x - beta * (z - At_y)
        x = x - beta * (z - At_y); // n x 1

        // stop when small update
        // if norm(x_old - x) < tol * norm(x_old)
        if((x_old - x).norm() < tol * x_old.norm())
            converged_main = true;

        if(n_iter >= max_iter)
            converged_main = true;
    } while(!converged_main);
}
