#include "util.h"

#if ENABLE_TRACE
static Tracer g_tracer;
int Tracer::sTraceFD = -1;
#endif

namespace Sophus {
using namespace Eigen;

Eigen::Matrix3d JacobianR(const Eigen::Vector3d& w) {
    Matrix3d Jr = Matrix3d::Identity();
    double theta = w.norm();
    if(theta < 1e-5) {
        return Jr;
    }
    else {
        Vector3d k = w.normalized(); // k is unit vector of w
        Matrix3d K = SO3d::hat(k);
        Jr = Matrix3d::Identity() - (1-cos(theta))/theta*K + (1-sin(theta)/theta)*K*K;
    }
    return Jr;
}

Eigen::Matrix3d JacobianRInv(const Eigen::Vector3d& w) {
    Matrix3d Jrinv = Matrix3d::Identity();
    double theta = w.norm();
    if(theta < 1e-5) {
        return Jrinv;
    }
    else {
        Vector3d k = w.normalized(); // k is unit vector of w
        Matrix3d K = SO3d::hat(k);
        Jrinv = Matrix3d::Identity() + 0.5 * SO3d::hat(w) +
                (1.0 + theta*(1.0+cos(theta))/(2.0*sin(theta)))*K*K;
    }
    return Jrinv;
}

Eigen::Matrix3d JacobianL(const Eigen::Vector3d& w) {
    return JacobianR(-w);
}

Eigen::Matrix3d JacobianLInv(const Eigen::Vector3d& w) {
    return JacobianRInv(-w);
}

}

namespace util {
ImagePyr Pyramid(const cv::Mat& img, int num_levels) {
    if(num_levels <= 1)
        throw cppt_exception("num_levels must > 1");

    ImagePyr image_pyr;
    image_pyr.resize(num_levels);
    image_pyr[0] = img;

    for(int i = 1; i < num_levels; ++i) {
        cv::resize(image_pyr[i-1], image_pyr[i], image_pyr[i-1].size()/2);
    }
    return image_pyr;
}
}
