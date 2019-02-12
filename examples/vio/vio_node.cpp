#include <fstream>
#include <queue>
#include <thread>
#include <condition_variable>
#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud.h>
#include <opencv2/core/eigen.hpp>
#include <sophus/se3.hpp>
#include <nav_msgs/Path.h>
#include <visualization_msgs/MarkerArray.h>
// catch ctrl+c signal
#include <signal.h>

#include "ros_utility.h"
#include "src/CameraPoseVisualization.h"
#include "system.h"
using namespace std;
using namespace message_filters;
using namespace sensor_msgs;

class Node {
public:
    using Measurements = vector<pair<pair<ImageConstPtr, ImageConstPtr>, vector<ImuConstPtr>>>;
    Node() : camera_pose_visual(1, 0, 0, 1)
    {
        camera_pose_visual.setScale(0.3);
        t_system = std::thread(&Node::SystemThread, this);
    }
    ~Node() {}

    void ReadFromNodeHandle(ros::NodeHandle& nh) {
        std::string config_file;
        config_file = readParam<std::string>(nh, "config_file");

        cv::FileStorage fs(config_file, cv::FileStorage::READ);
        fs["imu_topic"] >> imu_topic;
        fs["image_topic"] >> img_topic[0];
        fs["image_r_topic"] >> img_topic[1];
        fs["output_path"] >> log_filename;
        fs.release();

        system = std::make_shared<System>(config_file);
        system->SetDrawTrackingImgCallback(std::bind(&Node::PubTrackImg, this, std::placeholders::_1,
                                                     std::placeholders::_2, std::placeholders::_3));
        system->SetDrawMapPointCallback(std::bind(&Node::PubMapPoint, this, std::placeholders::_1,
                                                  std::placeholders::_2, std::placeholders::_3));
        system->SetDrawPoseCallback(std::bind(&Node::PubCurPose, this, std::placeholders::_1,
                                              std::placeholders::_2,std::placeholders::_3));
        system->SetDrawSlidingWindowCallback(std::bind(&Node::PubSlidingWindow, this, std::placeholders::_1,
                                                       std::placeholders::_2, std::placeholders::_3));
        system->SetDrawMarginMpsCallback(std::bind(&Node::PubMarginMps, this, std::placeholders::_1,
                                                   std::placeholders::_2, std::placeholders::_3));
    }

    void ImageCallback(const ImageConstPtr& img_msg, const ImageConstPtr& img_r_msg) {
        unique_lock<mutex> lock(m_buf);
        if(image_timestamp != -1.0f && std::abs(img_msg->header.stamp.toSec() - image_timestamp) > 10) { // 10 ms
            // reset system
            ROS_WARN_STREAM("System detect different image flow, trying to reset the system.");
            while(!img_buf.empty())
                img_buf.pop();

            while(!imu_buf.empty())
                imu_buf.pop();

            system->Reset();
        }
        img_buf.emplace(img_msg, img_r_msg);
        image_timestamp = img_msg->header.stamp.toSec();
        cv_system.notify_one();
    }

    void ImuCallback(const ImuConstPtr& imu_msg) {
        unique_lock<mutex> lock(m_buf);
        imu_buf.emplace(imu_msg);
        cv_system.notify_one();
    }

    Measurements GetMeasurements() {
        // The buffer mutex is locked before this function be called.
        Measurements measurements;

        while (1) {
            if (imu_buf.empty() || img_buf.empty())
                return measurements;

            double img_ts = img_buf.front().first->header.stamp.toSec();
            // catch the imu data before image_timestamp
            // ---------------^-----------^ image
            //                f           f+1
            // --x--x--x--x--x--x--x--x--x- imu
            //   f                       b
            // --o--o--o--o--o^-?---------- collect data in frame f

            // if ts(imu(b)) < ts(img(f)), wait imu data
            if (imu_buf.back()->header.stamp.toSec() < img_ts) {
                return measurements;
            }
            // if ts(imu(f)) > ts(img(f)), img data faster than imu data, drop the img(f)
            if (imu_buf.front()->header.stamp.toSec() > img_ts) {
                img_buf.pop();
                continue;
            }

            pair<ImageConstPtr, ImageConstPtr> img_msg = img_buf.front();
            img_buf.pop();

            vector<ImuConstPtr> IMUs;
            while (imu_buf.front()->header.stamp.toSec() < img_ts) {
                IMUs.emplace_back(imu_buf.front());
                imu_buf.pop();
            }
            // IMUs.emplace_back(imu_buf.front()); // ??
            measurements.emplace_back(img_msg, IMUs);
        }
    }

    void SystemThread() {
        while(1) {
            Measurements measurements;
            std::unique_lock<std::mutex> lock(m_buf);
            cv_system.wait(lock, [&] {
                return (measurements = GetMeasurements()).size() != 0;
            });
            lock.unlock();

            // TODO
            for(auto& meas : measurements) {
                ImageConstPtr img_msg = meas.first.first;
                ImageConstPtr img_msg_right = meas.first.second;
                vector<ImuConstPtr>& v_imu_msg = meas.second;
                double timestamp = img_msg->header.stamp.toSec();
                cv::Mat img_left, img_right;
                img_left = cv_bridge::toCvCopy(img_msg, "mono8")->image;
                img_right = cv_bridge::toCvCopy(img_msg_right, "mono8")->image;

                Eigen::VecVector3d v_gyr, v_acc;
                std::vector<double> v_imu_t;
                for(auto& imu_msg : v_imu_msg) {
                    v_imu_t.emplace_back(imu_msg->header.stamp.toSec());
                    v_gyr.emplace_back(imu_msg->angular_velocity.x, imu_msg->angular_velocity.y, imu_msg->angular_velocity.z);
                    v_acc.emplace_back(imu_msg->linear_acceleration.x, imu_msg->linear_acceleration.y, imu_msg->linear_acceleration.z);
                }

                system->Process(img_left, img_right, timestamp, v_gyr, v_acc, v_imu_t);
            }
        }
    }


    void PubTrackImg(const cv::Mat& track_img, uint64_t seq, double timestamp) {
        cv_bridge::CvImage track_cvimage;
        track_cvimage.header.seq = seq;
        track_cvimage.header.frame_id = "world";
        track_cvimage.header.stamp.fromSec(timestamp);
        track_cvimage.image = track_img;
        track_cvimage.encoding = sensor_msgs::image_encodings::BGR8;
        pub_track_img.publish(track_cvimage.toImageMsg());
    }

    void PubMapPoint(uint64_t seq, double timestamp, const Eigen::VecVector3d& mps) {
        sensor_msgs::PointCloud mps_msg;
        mps_msg.header.frame_id = "world";
        mps_msg.header.seq = seq;
        mps_msg.header.stamp.fromSec(timestamp);

        for(auto& x3Dw : mps) {
            geometry_msgs::Point32 point_marker;
            point_marker.x = x3Dw(0);
            point_marker.y = x3Dw(1);
            point_marker.z = x3Dw(2);
            mps_msg.points.emplace_back(point_marker);
        }
        pub_mappoints.publish(mps_msg);
    }

    void PubSlidingWindow(uint64_t seq, double timestamp, const std::vector<Sophus::SE3d>& v_Twc) {
        if(v_Twc.empty())
            return;

        { // print keyframe point
            visualization_msgs::Marker key_poses;
            key_poses.header.frame_id = "world";
            key_poses.header.seq = seq;
            key_poses.header.stamp.fromSec(timestamp);
            key_poses.ns = "keyframes";
            key_poses.type = visualization_msgs::Marker::SPHERE_LIST;
            key_poses.action = visualization_msgs::Marker::ADD;
            key_poses.pose.orientation.w = 1.0;
            key_poses.lifetime = ros::Duration();

            key_poses.id = 0;
            key_poses.scale.x = 0.05;
            key_poses.scale.y = 0.05;
            key_poses.scale.z = 0.05;
            key_poses.color.r = 1.0;
            key_poses.color.a = 1.0;

            for(auto& Twc : v_Twc) {
                Eigen::Vector3d twc = Twc.translation();
                geometry_msgs::Point pose_marker;
                pose_marker.x = twc(0);
                pose_marker.y = twc(1);
                pose_marker.z = twc(2);
                key_poses.points.emplace_back(pose_marker);
            }
            pub_keyframes.publish(key_poses);
        }
    }

    void PubCurPose(uint64_t seq, double timestamp, const Sophus::SE3d& Twc) {
        // public latest frame
        Eigen::Vector3d twc = Twc.translation();
        Eigen::Quaterniond qwc = Twc.so3().unit_quaternion();
        geometry_msgs::PoseStamped pose_stamped;
        pose_stamped.header.seq = seq;
        pose_stamped.header.frame_id = "world";
        pose_stamped.header.stamp.fromSec(timestamp);
        pose_stamped.pose.orientation.w = qwc.w();
        pose_stamped.pose.orientation.x = qwc.x();
        pose_stamped.pose.orientation.y = qwc.y();
        pose_stamped.pose.orientation.z = qwc.z();
        pose_stamped.pose.position.x = twc(0);
        pose_stamped.pose.position.y = twc(1);
        pose_stamped.pose.position.z = twc(2);
        path.poses.emplace_back(pose_stamped);
        pub_path.publish(path);

        // camera pose
        camera_pose_visual.reset();
        camera_pose_visual.add_pose(twc, qwc);
        camera_pose_visual.publish_by(pub_camera_pose, path.header);
    }

    void PubMarginMps(uint64_t seq, double timestamp, const Eigen::VecVector3d& margin_mps) {
        sensor_msgs::PointCloud margin_mps_msg;
        margin_mps_msg.header.frame_id = "world";
        margin_mps_msg.header.seq = seq;
        margin_mps_msg.header.stamp.fromSec(timestamp);

        for(auto& it : margin_mps) {
            geometry_msgs::Point32 p;
            p.x = it(0);
            p.y = it(1);
            p.z = it(2);
            margin_mps_msg.points.push_back(p);
        }
        pub_margin_mps.publish(margin_mps_msg);
    }

    string imu_topic;
    string img_topic[2];
    string log_filename;

    mutex m_buf;
    queue<ImuConstPtr> imu_buf;
    queue<pair<ImageConstPtr, ImageConstPtr>> img_buf;

    condition_variable cv_system;
    thread t_system;

    SystemPtr system;

    ros::Publisher pub_track_img;
    ros::Publisher pub_track_img_r;

    ros::Publisher pub_path;
    nav_msgs::Path path;
    ros::Publisher pub_camera_pose;
    ros::Publisher pub_keyframes;
    ros::Publisher pub_mappoints, pub_margin_mps;

    CameraPoseVisualization camera_pose_visual;

    double image_timestamp = -1.0f;
};

// global variable
Node node;
void sigint_handler(int s) {
    ROS_INFO_STREAM("logging trajectory to the file");
    std::ofstream fout(node.log_filename);
    if(!fout.is_open())
        exit(1);

    for(auto& pose : node.path.poses) {
        fout << pose.header.stamp << " ";
        fout << pose.pose.position.x << " " << pose.pose.position.y << " " <<
                pose.pose.position.z << " ";
        fout << pose.pose.orientation.x << " " << pose.pose.orientation.y << " "
             << pose.pose.orientation.z << " " << pose.pose.orientation.w << std::endl;
    }

    exit(1);
}

int main(int argc, char** argv) {
    ros::init(argc, argv, "cppt_vio", ros::init_options::NoSigintHandler);
//    ros::init(argc, argv, "cppt_vio");
    ros::NodeHandle nh("~");
    ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);

//    signal(SIGINT, sigint_handler);

    node.ReadFromNodeHandle(nh);

    message_filters::Subscriber<Image> sub_img[2] {{nh, node.img_topic[0], 100},
                                                   {nh, node.img_topic[1], 100}};
    TimeSynchronizer<Image, Image> sync(sub_img[0], sub_img[1], 100);
    sync.registerCallback(boost::bind(&Node::ImageCallback, &node, _1, _2));

    ros::Subscriber sub_imu = nh.subscribe(node.imu_topic, 2000, &Node::ImuCallback, &node,
                                           ros::TransportHints().tcpNoDelay());

    node.pub_track_img = nh.advertise<sensor_msgs::Image>("feature_img", 1000);
    node.pub_track_img_r = nh.advertise<sensor_msgs::Image>("feature_img_r", 1000);
    node.pub_path = nh.advertise<nav_msgs::Path>("path", 1000);
    node.path.header.frame_id = "world";
    node.pub_camera_pose = nh.advertise<visualization_msgs::MarkerArray>("camera_pose", 1000);
    node.pub_keyframes = nh.advertise<visualization_msgs::Marker>("keyframes", 1000);
    node.pub_mappoints = nh.advertise<sensor_msgs::PointCloud>("mappoints", 1000);
    node.pub_margin_mps = nh.advertise<sensor_msgs::PointCloud>("margin_mps", 1000);
    ROS_INFO_STREAM("Player is ready.");

    ros::spin();
    return 0;
}
