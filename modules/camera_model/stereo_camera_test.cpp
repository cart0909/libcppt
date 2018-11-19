#include "stereo_camera.h"
#include <gtest/gtest.h>

static int image_width = 752;
static int image_height = 480;
static double fx = 458.654;
static double fy = 457.296;
static double cx = 367.215;
static double cy = 248.375;
static double k1 = -0.28340811;
static double k2 = 0.07395907;
static double p1 = 0.00019359;
static double p2 = 1.76187114e-05;

static Eigen::Matrix4d T_BC0;

static double fx_r = 457.587;
static double fy_r = 456.134;
static double cx_r = 379.999;
static double cy_r = 255.238;
static double k1_r = -0.28368365;
static double k2_r = 0.07451284;
static double p1_r = -0.00010473;
static double p2_r = -3.55590700e-05;

static Eigen::Matrix4d T_BC1;

static Eigen::Matrix4d T_BI = Eigen::Matrix4d::Identity();

TEST(stereo_camera, extrinsic) {
    T_BC0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                           0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                          -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                           0.0, 0.0, 0.0, 1.0;

    T_BC1 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                           0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                          -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                           0.0, 0.0, 0.0, 1.0;

    PinholeCameraPtr left_cam(new PinholeCamera("L", image_width, image_height, fx, fy, cx, cy, k1, k2, p1, p2));
    PinholeCameraPtr right_cam(new PinholeCamera("R", image_width, image_height, fx_r, fy_r, cx_r, cy_r,
                                                 k1_r, k2_r, p1_r, p2_r));
    ImuSensorPtr imu(new ImuSensor());

    left_cam->Tbs = Sophus::SE3d(T_BC0);
    right_cam->Tbs = Sophus::SE3d(T_BC1);

    StereoCameraPtr stereo(new StereoCamera(imu, left_cam, right_cam));

    Eigen::Matrix4d Trl = stereo->mTij[StereoCamera::I_RC][StereoCamera::I_LC].matrix();
    Eigen::Matrix4d Tbl = stereo->mTij[StereoCamera::I_IMU][StereoCamera::I_LC].matrix();
    Eigen::Matrix4d Tbr = stereo->mTij[StereoCamera::I_IMU][StereoCamera::I_RC].matrix();
    Eigen::Matrix4d Tlr = stereo->mTij[StereoCamera::I_LC][StereoCamera::I_RC].matrix();
    Eigen::Matrix4d Tlb = stereo->mTij[StereoCamera::I_LC][StereoCamera::I_IMU].matrix();
    Eigen::Matrix4d Trb = stereo->mTij[StereoCamera::I_RC][StereoCamera::I_IMU].matrix();
    Eigen::Matrix4d TC1C0 = T_BC1.inverse() * T_BC0;

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(TC1C0(i, j), Trl(i, j));
        }
    }

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(T_BC0(i, j), Tbl(i, j));
        }
    }

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(T_BC1(i, j), Tbr(i, j));
        }
    }

    Eigen::Matrix4d TC0C1, TC0B, TC1B;
    TC0C1 = TC1C0.inverse();
    TC0B = T_BC0.inverse();
    TC1B = T_BC1.inverse();

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(TC0C1(i, j), Tlr(i, j));
        }
    }

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(TC0B(i, j), Tlb(i, j));
        }
    }

    for(int i = 0; i < 3; ++i) {
        for(int j = 0; j < 4; ++j) {
            EXPECT_FLOAT_EQ(TC1B(i, j), Trb(i, j));
        }
    }
}

TEST(stereo_camera, triangulate) {
    T_BC0 << 0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975,
                           0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768,
                          -0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949,
                           0.0, 0.0, 0.0, 1.0;

    T_BC1 << 0.0125552670891, -0.999755099723, 0.0182237714554, -0.0198435579556,
                           0.999598781151, 0.0130119051815, 0.0251588363115, 0.0453689425024,
                          -0.0253898008918, 0.0179005838253, 0.999517347078, 0.00786212447038,
                           0.0, 0.0, 0.0, 1.0;

    PinholeCameraPtr left_cam(new PinholeCamera("L", image_width, image_height, fx, fy, cx, cy, k1, k2, p1, p2));
    PinholeCameraPtr right_cam(new PinholeCamera("R", image_width, image_height, fx_r, fy_r, cx_r, cy_r,
                                                 k1_r, k2_r, p1_r, p2_r));
    ImuSensorPtr imu(new ImuSensor());

    left_cam->Tbs = Sophus::SE3d(T_BC0);
    right_cam->Tbs = Sophus::SE3d(T_BC1);

    StereoCameraPtr stereo(new StereoCamera(imu, left_cam, right_cam));

    Eigen::Vector3d Xl, Xr;
    Xl << -0.281348,
          0.321307,
           0.57979;

    Xr = stereo->mTij[1][0].rotationMatrix() * Xl + stereo->mTij[1][0].translation();

    Eigen::Vector2d x, xr;
    stereo->mpCamera[0]->Project(Xl, x);
    stereo->mpCamera[1]->Project(Xr, xr);

    Eigen::Vector3d ray_ll, ray_rr;
    stereo->mpCamera[0]->BackProject(x, ray_ll);
    stereo->mpCamera[1]->BackProject(xr, ray_rr);
    Eigen::Vector3d tXl = stereo->Triangulate(ray_ll, ray_rr);

    for(int i = 0; i < 3; ++i)
        EXPECT_FLOAT_EQ(Xl(i), tXl(i));
}
