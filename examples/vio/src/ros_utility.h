#include <ros/ros.h>
#include <string>

template<typename T>
T readParam(ros::NodeHandle &n, const std::string& name) {
    T ans;
    if(n.getParam(name, ans)) {
        ROS_INFO_STREAM("Loaded " << name << ": " << ans);
    }
    else {
        ROS_ERROR_STREAM("Failed to load " << name);
        n.shutdown();
    }
    return ans;
}
