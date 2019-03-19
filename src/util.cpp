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
        throw std::runtime_error("num_levels must > 1");

    ImagePyr image_pyr;
    image_pyr.resize(num_levels);
    image_pyr[0] = img;

    for(int i = 1; i < num_levels; ++i) {
        cv::resize(image_pyr[i-1], image_pyr[i], image_pyr[i-1].size()/2);
    }
    return image_pyr;
}

double f2fLineSegmentOverlap(Eigen::Vector2d spl_obs, Eigen::Vector2d epl_obs, Eigen::Vector2d spl_proj, Eigen::Vector2d epl_proj){
    double overlap = 1.f;
    if( std::abs(spl_obs(0) - epl_obs(0)) < 1.0 )         // vertical lines
    {
        // line equations
        Eigen::Vector2d l = epl_obs - spl_obs;

        // intersection points
        Eigen::Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_obs(0), spl_proj(1);
        epl_proj_line << epl_obs(0), epl_proj(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(1)-spl_obs(1)) / l(1);
        double lambda_e = (epl_proj_line(1)-spl_obs(1)) / l(1);

        double lambda_min = std::min(lambda_s,lambda_e);
        double lambda_max = std::max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else if( std::abs(spl_obs(1)-epl_obs(1)) < 1.0 )    // horizontal lines (previously removed)
    {
        // line equations
        Eigen::Vector2d l = epl_obs - spl_obs;

        // intersection points
        Eigen::Vector2d spl_proj_line, epl_proj_line;
        spl_proj_line << spl_proj(0), spl_obs(1);
        epl_proj_line << epl_proj(0), epl_obs(1);

        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = std::min(lambda_s,lambda_e);
        double lambda_max = std::max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }
    else                                            // non-degenerate cases
    {
        // line equations
        Eigen::Vector2d l = epl_obs - spl_obs;
        double a = spl_obs(1)-epl_obs(1);
        double b = epl_obs(0)-spl_obs(0);
        double c = spl_obs(0)*epl_obs(1) - epl_obs(0)*spl_obs(1);

        // intersection points
        Eigen::Vector2d spl_proj_line, epl_proj_line;
        double lxy = 1.f / (a*a+b*b);
#if 0
        spl_proj_line << ( b*( b*spl_proj(0)-a*spl_proj(1))-a*c ) * lxy,
                ( a*(-b*spl_proj(0)+a*spl_proj(1))-b*c ) * lxy;

        epl_proj_line << ( b*( b*epl_proj(0)-a*epl_proj(1))-a*c ) * lxy,
                ( a*(-b*epl_proj(0)+a*epl_proj(1))-b*c ) * lxy;
#else
        spl_proj_line << spl_proj(0) + (-a * (a*spl_proj(0) + spl_proj(1)*b +c ) * lxy),
                spl_proj(1) + (-b * (a*spl_proj(0) + spl_proj(1)*b +c ) * lxy);

        epl_proj_line << epl_proj(0) + (-a * (a*epl_proj(0) + epl_proj(1)*b +c ) * lxy),
                epl_proj(1) + (-b * (a*epl_proj(0) + epl_proj(1)*b +c ) * lxy);
#endif
        // estimate overlap in function of lambdas
        double lambda_s = (spl_proj_line(0)-spl_obs(0)) / l(0);
        double lambda_e = (epl_proj_line(0)-spl_obs(0)) / l(0);

        double lambda_min = std::min(lambda_s,lambda_e);
        double lambda_max = std::max(lambda_s,lambda_e);

        if( lambda_min < 0.f && lambda_max > 1.f )
            overlap = 1.f;
        else if( lambda_max < 0.f || lambda_min > 1.f )
            overlap = 0.f;
        else if( lambda_min < 0.f )
            overlap = lambda_max;
        else if( lambda_max > 1.f )
            overlap = 1.f - lambda_min;
        else
            overlap = lambda_max - lambda_min;

    }

    return overlap;

}
} //namespace util

