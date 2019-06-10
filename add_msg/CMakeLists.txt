cmake_minimum_required(VERSION 2.8.3)
project(add_msg)

find_package(catkin REQUIRED COMPONENTS 
             roscpp 
             rospy 
             message_generation 
             std_msgs 
             geometry_msgs)

add_message_files(
  FILES
  ImuPredict.msg
)

generate_messages(
 DEPENDENCIES
 geometry_msgs
)

catkin_package(
CATKIN_DEPENDS message_runtime
)
