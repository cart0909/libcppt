#pragma once
#include "util.h"
#include "LocalCartesian.hpp"
#include "mutex"
#include "thread"
#include "ros/ros.h"
#include "nav_msgs/Path.h"
class GlobalOptimize{
public:
    GlobalOptimize();
    ~GlobalOptimize();
    void inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy);

    //convert wvio_T_body (local)pose to global pose and save both info.
    void inputOdom(double t, Sophus::SE3d wvio_T_body);
    nav_msgs::Path global_path;
    void getGlobalOdom(Sophus::SE3d& wgps_T_bodyCur);
    void getWGPS_T_WVIO(Sophus::SE3d& WGPS_T_WVIO_);
    void getGPSXYZ(double t, std::vector<double>& xyz);
    void AddPoseGlobalPath(double t, Sophus::SE3d& wgps_Twvio);
    bool USEGPS = false;
private:
    void GPS2XYZ(double latitude, double longitude, double altitude, double* xyz);
    void GPSoptimize();
    void UpdateAllGlobalPath();
    void RemoveOutWindowPose();
    // format t, tx,ty,tz,qw,qx,qy,qz
    std::map<double, Sophus::SE3d> localPoseMap;
    std::map<double, Sophus::SE3d> globalPoseMap;
    //format t, tx,ty,tz poseAccuracy
    std::map<double, std::vector<double>> GPSPositionMap;
    bool initGPS;
    bool newGPS;
    GeographicLib::LocalCartesian geoConverter;
    std::mutex mPoseMap;
    Sophus::SE3d WGPS_T_WVIO;
    Sophus::SE3d WGPS_T_body_cur;
    std::thread threadOpt;
    Eigen::Quaterniond I;
    //FixWindowSize for global optimization
    int WindowSize = 50;
    //Using Gps threshold for judge the gps and vio fusion condition.
    double useGps_TH = 0.2;
};

SMART_PTR(GlobalOptimize);
