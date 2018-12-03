#include "simple_stereo_camera.h"

SimpleStereoCam::SimpleStereoCam(const Sophus::SE3d& Tbs_, int width_, int height_,
                                 double f_, double cx_, double cy_, double b_,
                                 const cv::Mat& M1l_, const cv::Mat& M2l_,
                                 const cv::Mat& M1r_, const cv::Mat& M2r_)
    : SensorBase(Tbs_), width(width_), height(height_), f(f_), inv_f(1.0f/f),
      cx(cx_), cy(cy_), b(b_), bf(b*f),
      M1l(M1l_.clone()), M2l(M2l_.clone()), M1r(M1r_.clone()), M2r(M2r_.clone())
{

}

SimpleStereoCam::~SimpleStereoCam() {}
