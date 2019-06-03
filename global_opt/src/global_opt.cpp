#include "global_opt.h"
#include "globalfactor.h"
#include <vector>
#include <iostream>
GlobalOptimize::GlobalOptimize(){
    initGPS = false;
    newGPS = false;
    I.setIdentity();
    WGPS_T_WVIO.so3().setQuaternion(I);
    WGPS_T_body_cur.so3().setQuaternion(I);
    threadOpt = std::thread(&GlobalOptimize::GPSoptimize, this);
}
GlobalOptimize::~GlobalOptimize(){

}

void GlobalOptimize::inputGPS(double t, double latitude, double longitude, double altitude, double posAccuracy){
    double xyz[3];
    GPS2XYZ(latitude, longitude, altitude, xyz);
    std::vector<double> tmp{xyz[0], xyz[1], xyz[2], posAccuracy};
    GPSPositionMap[t] = tmp;
    newGPS = true;
}

void GlobalOptimize::inputOdom(double t, Sophus::SE3d wvio_T_body){
    mPoseMap.lock();
    localPoseMap[t] = wvio_T_body;
    globalPoseMap[t] = WGPS_T_WVIO * wvio_T_body;
    WGPS_T_body_cur = WGPS_T_WVIO * wvio_T_body;

    //TODO:: push pose of WGPS_T_body_cur
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(t);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = WGPS_T_body_cur.translation().x();
    pose_stamped.pose.position.y = WGPS_T_body_cur.translation().y();
    pose_stamped.pose.position.z = WGPS_T_body_cur.translation().z();
    pose_stamped.pose.orientation.x = WGPS_T_body_cur.so3().unit_quaternion().x();
    pose_stamped.pose.orientation.y = WGPS_T_body_cur.so3().unit_quaternion().y();
    pose_stamped.pose.orientation.z = WGPS_T_body_cur.so3().unit_quaternion().z();
    pose_stamped.pose.orientation.w = WGPS_T_body_cur.so3().unit_quaternion().w();
    global_path.header = pose_stamped.header;
    global_path.poses.push_back(pose_stamped);

    mPoseMap.unlock();
}

void GlobalOptimize::getWGPS_T_WVIO(Sophus::SE3d& WGPS_T_WVIO_){
    WGPS_T_WVIO_ = WGPS_T_WVIO;
}

void GlobalOptimize::getGlobalOdom(Sophus::SE3d& wgps_T_bodyCur){
    wgps_T_bodyCur = WGPS_T_body_cur;
}

void GlobalOptimize::getGPSXYZ(double t, std::vector<double>& xyz){
    xyz.resize(4);
    xyz = GPSPositionMap[t];
}

void GlobalOptimize::GPS2XYZ(double latitude, double longitude, double altitude, double* xyz){
    if(!initGPS)
    {
        geoConverter.Reset(latitude, longitude, altitude);
        initGPS = true;
    }
    geoConverter.Forward(latitude, longitude, altitude, xyz[0], xyz[1], xyz[2]);

}


void GlobalOptimize::UpdateAllGlobalPath()
{
    global_path.poses.clear();
    std::map<double, Sophus::SE3d>::iterator iter;
    for (iter = globalPoseMap.begin(); iter != globalPoseMap.end(); iter++)
    {
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.stamp = ros::Time(iter->first);
        pose_stamped.header.frame_id = "world";
        pose_stamped.pose.position.x = iter->second.translation().x();
        pose_stamped.pose.position.y = iter->second.translation().y();
        pose_stamped.pose.position.z = iter->second.translation().z();
        pose_stamped.pose.orientation.w = iter->second.so3().unit_quaternion().w();
        pose_stamped.pose.orientation.x = iter->second.so3().unit_quaternion().x();
        pose_stamped.pose.orientation.y = iter->second.so3().unit_quaternion().y();
        pose_stamped.pose.orientation.z = iter->second.so3().unit_quaternion().z();
        global_path.poses.push_back(pose_stamped);
    }
}

void GlobalOptimize::AddPoseGlobalPath(double t, Sophus::SE3d& wgps_Twvio)
{
    //Add curr wgps_Twvio to global_path.
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.header.stamp = ros::Time(t);
    pose_stamped.header.frame_id = "world";
    pose_stamped.pose.position.x = wgps_Twvio.translation().x();
    pose_stamped.pose.position.y = wgps_Twvio.translation().y();
    pose_stamped.pose.position.z = wgps_Twvio.translation().z();
    pose_stamped.pose.orientation.w = wgps_Twvio.so3().unit_quaternion().w();
    pose_stamped.pose.orientation.x = wgps_Twvio.so3().unit_quaternion().x();
    pose_stamped.pose.orientation.y = wgps_Twvio.so3().unit_quaternion().y();
    pose_stamped.pose.orientation.z = wgps_Twvio.so3().unit_quaternion().z();
    global_path.poses.push_back(pose_stamped);
}

void GlobalOptimize::RemoveOutWindowPose(){
    std::map<double, Sophus::SE3d>::iterator iter_global, iter_local;
    int length = globalPoseMap.size();
    int num_remove = length - WindowSize;
    int iter_id = 0;
    iter_global = globalPoseMap.begin();
    //remove globalmap
    while (iter_global != globalPoseMap.end()) {
        iter_id++;
        if (iter_id < num_remove)
            iter_global = globalPoseMap.erase(iter_global);
        else
            break;
    }
    //remove localMap
    iter_id = 0;
    iter_local = localPoseMap.begin();
    while (iter_local != localPoseMap.end()) {
        iter_id++;
        if (iter_id < num_remove)
            iter_local = localPoseMap.erase(iter_local);
        else
            break;
    }
}

void GlobalOptimize::GPSoptimize(){

    while (true) {
        if(newGPS)
        {
            newGPS = false;
            ceres::Problem problem;
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
            options.max_num_iterations = 5;
            ceres::Solver::Summary summary;
            ceres::LossFunction *loss_function;
            loss_function = new ceres::HuberLoss(1.0);
            ceres::LocalParameterization* local_parameterization = new autodiff::LocalParameterizationSO3();

            mPoseMap.lock();
            //add vertex
            int length = localPoseMap.size();
            std::map<double, Sophus::SE3d>::iterator iter;
            double paraP_gps_b[3 * length];
            double paraQ_gps_b[4 * length];
            iter = globalPoseMap.begin();
            for (int i = 0; i < length; i++, iter++)
            {
                size_t p_idx_pose = i * 3;
                size_t q_idx_pose = i * 4;
                std::memcpy(paraQ_gps_b + q_idx_pose, iter->second.data(), sizeof(double) * 4);
                std::memcpy(paraP_gps_b + p_idx_pose, iter->second.translation().data(), sizeof(double) * 3);
                problem.AddParameterBlock(paraP_gps_b + p_idx_pose, 3);
                problem.AddParameterBlock(paraQ_gps_b + q_idx_pose, 4, local_parameterization);
            }


            //add VIO edge info
            std::map<double, Sophus::SE3d>::iterator iterVIO, iterVIONext;
            std::map<double, std::vector<double>>::iterator iterGPS;
            int i = 0;

            for (iterVIO = localPoseMap.begin(); iterVIO != localPoseMap.end(); iterVIO++, i++)
            {
                //vio factor
                iterVIONext = iterVIO;
                iterVIONext++;
                if(iterVIONext != localPoseMap.end())
                {
                    Sophus::SE3d wTi = iterVIO->second;
                    Sophus::SE3d wTj = iterVIONext->second;
                    Sophus::SE3d iTj = wTi.inverse() * wTj;
                    ceres::CostFunction* vio_function = RelativeRTError::Create(iTj, 0.1, 0.01);
                    problem.AddResidualBlock(vio_function, NULL, paraP_gps_b + 3 * i,
                                             paraQ_gps_b + 4 * i,
                                             paraP_gps_b + 3 * (i + 1), paraQ_gps_b + 4 * (i + 1));
                }
                //gps factor
                double t = iterVIO->first;
                iterGPS = GPSPositionMap.find(t);
                if (iterGPS != GPSPositionMap.end())
                {
                    ceres::CostFunction* gps_function = TError::Create(iterGPS->second[0], iterGPS->second[1],
                            iterGPS->second[2], iterGPS->second[3]);
                    problem.AddResidualBlock(gps_function, loss_function, paraP_gps_b + 3 * i);
                }
            }

            ceres::Solve(options, &problem, &summary);

            //For estimate the motion variation of the wgpsTwboyd before and after
            Eigen::Vector3d t_motion_sum;
            t_motion_sum.setZero();

            //Update Global pose and find the WVIO_T_WVIO
            iter = globalPoseMap.begin();
            for (int j = 0; j < length; j++, iter++)
            {
                size_t p_idx_pose = j * 3;
                size_t q_idx_pose = j * 4;
                iterGPS = GPSPositionMap.find(iter->first);
                std::memcpy(iter->second.data(), paraQ_gps_b + q_idx_pose, sizeof(double) * 4);
                std::memcpy(iter->second.data() + 4, paraP_gps_b + p_idx_pose, sizeof(double) * 3);
                Sophus::SE3d wgpsTb_new = iter->second;

                Eigen::Vector3d error;
                error[0] = fabs(wgpsTb_new.translation().x() - iterGPS->second[0]);
                error[1] = fabs(wgpsTb_new.translation().y() - iterGPS->second[1]);
                error[2] = fabs(wgpsTb_new.translation().z() - iterGPS->second[2]);
                t_motion_sum = t_motion_sum + error;

                if(j == length - 1){
                    //Estimate the WGPS_T_WVIO
                    double t = iter->first;
                    Sophus::SE3d WVIO_T_body = localPoseMap[t];
                    Sophus::SE3d WGPS_T_body = globalPoseMap[t];
                    WGPS_T_WVIO = WGPS_T_body * WVIO_T_body.inverse();
                }
            }

            //remove map info
            if(length > WindowSize){
                RemoveOutWindowPose();
                if((t_motion_sum * 1.0 / length).norm() < useGps_TH && USEGPS != true){
                    std::cout << "USEGPS SUCESS" <<std::endl;
                    USEGPS = true;
                }
            }
            //Update global path for rviz
            UpdateAllGlobalPath();
            mPoseMap.unlock();

        }
        //delay 2000
        std::chrono::milliseconds dura(2000);
        std::this_thread::sleep_for(dura);
    }

    return;
}
