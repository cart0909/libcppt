#include <gtest/gtest.h>
#include "pinhole_camera.h"

static int image_width = 752;
static int image_height = 480;
static double fx = 4.616e+02;
static double fy = 4.603e+02;
static double cx = 3.630e+02;
static double cy = 2.481e+02;
static double k1 = -2.917e-01;
static double k2 = 8.228e-02;
static double p1 = 5.333e-05;
static double p2 = -1.578e-04;

static PinholeCameraPtr camera(new PinholeCamera("euroc", image_width, image_height,
                                          fx, fy, cx, cy, k1, k2, p1, p2));
static PinholeCameraPtr virtual_camera(new PinholeCamera("euroc_undistort", image_width, image_height,
                                                  fx, fy, cx, cy));

static cv::Mat K = (cv::Mat_<double>(3, 3) << fx, 0, cx,
                                       0, fy, cy,
                                       0,  0,  1);

static cv::Mat D = (cv::Mat_<double>(4, 1) << k1, k2, p1, p2);

TEST(pinhole_camera, project) {
    double x = 0.24, y = -0.17, z = 0.87;
    std::vector<cv::Point3d> pt3ds {{x,y,z}};
    std::vector<cv::Point2d> pt2ds;
    cv::projectPoints(pt3ds, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K, D, pt2ds);

    Eigen::Vector2d u;
    camera->Project(Eigen::Vector3d(x, y, z), u);
    EXPECT_FLOAT_EQ(pt2ds[0].x, u(0));
    EXPECT_FLOAT_EQ(pt2ds[0].y, u(1));
}

TEST(pinhole_camera, Jacobian) {
    double x = 0.24, y = -0.17, z = 0.87;
    std::vector<cv::Point3d> pt3ds {{x,y,z}};
    std::vector<cv::Point2d> pt2ds;
    cv::Mat cvJ;
    cv::projectPoints(pt3ds, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K, D, pt2ds, cvJ);
    cvJ = cvJ.colRange(3, 6);

    Eigen::Vector2d u;
    Eigen::Matrix<double, 2, 3> eigenJ;
    camera->Project(Eigen::Vector3d(x, y, z), u, eigenJ);

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(cvJ.at<double>(i, j), eigenJ(i, j));
        }
    }
}

TEST(pinhole_camera, project_undistort) {
    double x = 0.24, y = -0.17, z = 0.87;
    std::vector<cv::Point3d> pt3ds {{x,y,z}};
    std::vector<cv::Point2d> pt2ds;
    cv::projectPoints(pt3ds, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K, cv::noArray(), pt2ds);

    Eigen::Vector2d u;
    virtual_camera->Project(Eigen::Vector3d(x, y, z), u);
    EXPECT_FLOAT_EQ(pt2ds[0].x, u(0));
    EXPECT_FLOAT_EQ(pt2ds[0].y, u(1));
}

TEST(pinhole_camera, Jacobian_undistort) {
    double x = 0.24, y = -0.17, z = 0.87;
    std::vector<cv::Point3d> pt3ds {{x,y,z}};
    std::vector<cv::Point2d> pt2ds;
    cv::Mat cvJ;
    cv::projectPoints(pt3ds, cv::Mat::zeros(3, 1, CV_64F), cv::Mat::zeros(3, 1, CV_64F), K, cv::noArray(), pt2ds, cvJ);
    cvJ = cvJ.colRange(3, 6);

    Eigen::Vector2d u;
    Eigen::Matrix<double, 2, 3> eigenJ;
    virtual_camera->Project(Eigen::Vector3d(x, y, z), u, eigenJ);

    for(int i = 0; i < 2; ++i) {
        for(int j = 0; j < 3; ++j) {
            EXPECT_FLOAT_EQ(cvJ.at<double>(i, j), eigenJ(i, j));
        }
    }
}

TEST(pinhole_camera, back_project) {
    double u = 377, v = 238;
    std::vector<cv::Point2d> pt2d {{u, v}};
    std::vector<cv::Point2d> pt2d_undistort;
    cv::undistortPoints(pt2d, pt2d_undistort, K, D, cv::noArray());

    Eigen::Vector3d X;
    camera->BackProject(Eigen::Vector2d(u, v), X);

    EXPECT_FLOAT_EQ(pt2d_undistort[0].x, X(0));
    EXPECT_FLOAT_EQ(pt2d_undistort[0].y, X(1));
}
