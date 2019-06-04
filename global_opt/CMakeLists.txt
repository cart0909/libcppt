cmake_minimum_required(VERSION 2.8.3)
project(global_opt)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DEIGEN_DONT_PARALLELIZE")
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

find_package(Eigen3 REQUIRED)
set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
message("EIGEN3_INCLUDE_DIR:" ${EIGEN3_INCLUDE_DIR})
find_package(OpenCV REQUIRED)
set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${OpenCV_LIBS})
message("OpenCV_INCLUDE_DIRS:" ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBS:" ${OpenCV_LIBS})
find_package(Ceres REQUIRED)
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${CERES_LIBRARIES})
message("CERES_LIBRARIES:" ${CERES_LIBRARIES})
find_package(GTest REQUIRED)
set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${GTEST_INCLUDE_DIRS})
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${GTEST_LIBRARIES})
message("GTEST_INCLUDE_DIRS:" ${GTEST_INCLUDE_DIRS})
message("GTEST_LIBRARIES:" ${GTEST_LIBRARIES})

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  sensor_msgs
  roscpp
  rospy
  rosbag
  std_msgs
  image_transport
  cv_bridge
  tf
  cppt
)
 
set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${catkin_LIBRARIES})

add_subdirectory(./ThirdParty/GeographicLib/)
include_directories(
  ./ThirdParty/GeographicLib/include/
  ${THRID_PARTY_INCLUDE_DIRS} 
)

catkin_package(
  CATKIN_DEPENDS 
  nav_msgs 
  roscpp 
  rospy 
  std_msgs
  DEPENDS EIGEN3 
  cppt
  tf
  INCLUDE_DIRS ./ThirdParty/GeographicLib/include/
)

add_executable(global_opt_node
               src/global_opt_node.cpp
               src/global_opt.cpp
               src/globalfactor.cpp)
target_link_libraries(global_opt_node libGeographic_lib ${THRID_PARTY_LIBS})

