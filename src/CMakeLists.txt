cmake_minimum_required(VERSION 2.8.11)
project(cppt)

# Cross Compile
if ("$ENV{CROSS_COMPILE_FLAG}" STREQUAL "XU4")
  message("Cross Compile For XU4")
  SET(CMAKE_SYSTEM_NAME Linux)
  SET(CMAKE_SYSTEM_PROCESSOR armv7l)
  SET(CMAKE_C_COMPILER /usr/bin/arm-linux-gnueabihf-gcc)
  SET(CMAKE_CXX_COMPILER /usr/bin/arm-linux-gnueabihf-g++)
  SET(CMAKE_BUILD_TYPE Release)
endif()

# If on OSX, force cmake to use gcc-6/5 instead of the built-in gcc (AppleClang)
# The compiler overriden has to happen before calling "project"
if (APPLE)
  if (EXISTS /usr/local/bin/g++-6)
    message("Override CXX compiler to /usr/local/bin/g++-6!")
    set(CMAKE_C_COMPILER   /usr/local/bin/gcc-6)
    set(CMAKE_CXX_COMPILER /usr/local/bin/g++-6)
  elseif (EXISTS /usr/local/bin/g++-5)
    message("Override CXX compiler to /usr/local/bin/g++-5!")
    set(CMAKE_C_COMPILER   /usr/local/bin/gcc-5)
    set(CMAKE_CXX_COMPILER /usr/local/bin/g++-5)
  elseif (EXISTS /usr/local/bin/g++-7)
    message("Override CXX compiler to /usr/local/bin/g++-7!")
    set(CMAKE_C_COMPILER   /usr/local/bin/gcc-7)
    set(CMAKE_CXX_COMPILER /usr/local/bin/g++-7)
  else ()
    message(FATAL_ERROR "Cannot find GNU gcc on Mac!")
  endif()
  set(CMAKE_MACOSX_RPATH 0)
endif()

# Configure cxx flags
if(CYGWIN)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11 -msse2 -Wno-unused-result")
else()
# NEON or SSE
if((CMAKE_SYSTEM_PROCESSOR MATCHES "^arm"))
  set(CMAKE_CXX_FLAGS "-D__ARM_NEON__ -DENABLE_NEON -mfloat-abi=hard -mfpu=neon -Ofast -std=c++11 -Wno-unused-result")
  message("Enabling neon for armv7 " ${CMAKE_CXX_FLAGS})
elseif (CMAKE_SYSTEM_PROCESSOR MATCHES "^aarch64")
  set(CMAKE_CXX_FLAGS "-D__ARM_NEON__ -DENABLE_NEON -Ofast -std=c++11 -Wno-unused-result")
  message("Detected armv8 " ${CMAKE_CXX_FLAGS})
else()
 # SSE4.1 seems to be not supported by GCC 4.8.4
 # TODO(mingyu): See if we still need sse4.1
 message(STATUS "Enable SSE2 and SSE4.1")
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -msse2 -msse4.1 -Wno-unused-result")
 # Export compile_commands.json
 set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
endif()
endif()

# Suppress warnings for deprecated declarations for GCC 5 and above
if (CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
 set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")
endif()

# enable openmp flag
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -DEIGEN_DONT_PARALLELIZE")

# set relase build ofast
set(CMAKE_CXX_FLAGS_RELEASE "-Ofast")

# Debug / Release / RelWithDebInfo
set(CMAKE_BUILD_TYPE Release)

# Thirdparty dependencies
# If CFG_VIEW is turned on, need to install glut (freeglut3-dev) and glew (libglew-dev)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

# add 3rd party
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

# add ros
find_package(catkin REQUIRED COMPONENTS
roscpp
roslib
cmake_modules
)

set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${catkin_LIBRARIES})

catkin_package(
DEPENDS
CATKIN_DEPENDS roscpp roslib
INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
LIBRARIES ${PROJECT_NAME}
)

# DBoW2
set(DBOW2_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/3rdParty/DBoW2)
set(DBOW2_SRC_FILES
${DBOW2_INCLUDE_DIRS}/VocabularyBinary.cpp
${DBOW2_INCLUDE_DIRS}/DBoW2/BowVector.cpp
${DBOW2_INCLUDE_DIRS}/DBoW2/FBrief.cpp
${DBOW2_INCLUDE_DIRS}/DBoW2/FeatureVector.cpp
${DBOW2_INCLUDE_DIRS}/DBoW2/FORB.cpp
${DBOW2_INCLUDE_DIRS}/DBoW2/QueryResults.cpp
${DBOW2_INCLUDE_DIRS}/DBoW2/ScoringObject.cpp
${DBOW2_INCLUDE_DIRS}/DUtils/Random.cpp
${DBOW2_INCLUDE_DIRS}/DUtils/Timestamp.cpp
${DBOW2_INCLUDE_DIRS}/DVision/BRIEF.cpp
)

# include
include_directories(
${THRID_PARTY_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
)

add_library(${PROJECT_NAME} SHARED
util.cpp
camera.cpp
feature_tracker.cpp
stereo_matcher.cpp
config_loader.cpp
system.cpp
backend.cpp
converter.cpp
relocalization.cpp
pose_faster.cpp
line_tracker.cpp
line_stereo_matcher.cpp
pl_system.cpp
pl_backend.cpp
rgbd_system.cpp
rgbd_backend.cpp
thread_pool_mgr.cpp
ceres/local_parameterization_so3.cpp
ceres/local_parameterization_se3.cpp
ceres/local_parameterization_line.cpp
ceres/projection_factor.cpp
ceres/projection_td_factor.cpp
ceres/line_projection_factor.cpp
vins/marginalization_factor.cpp
vins/integration_base.cpp
vins/imu_factor.cpp
${DBOW2_SRC_FILES}
)
target_link_libraries(${PROJECT_NAME} ${THRID_PARTY_LIBS})
