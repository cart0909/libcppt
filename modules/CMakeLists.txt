cmake_minimum_required(VERSION 2.8.11)
project(cppt_modules)

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

# Debug / Release / RelWithDebInfo
set(CMAKE_BUILD_TYPE Release)

# Thirdparty dependencies
# If CFG_VIEW is turned on, need to install glut (freeglut3-dev) and glew (libglew-dev)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/../cmake")

# add 3rd party
find_package(Eigen3 3.3.5 REQUIRED)
set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIR})
message("EIGEN3_INCLUDE_DIR:" ${EIGEN3_INCLUDE_DIR})
find_package(OpenCV 3.4.3 REQUIRED)
set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${OpenCV_LIBS})
message("OpenCV_INCLUDE_DIRS:" ${OpenCV_INCLUDE_DIRS})
message("OpenCV_LIBS:" ${OpenCV_LIBS})
find_package(Ceres REQUIRED)
set(THRID_PARTY_LIBS ${THRID_PARTY_LIBS} ${CERES_LIBRARIES})
message("CERES_LIBRARIES:" ${CERES_LIBRARIES})

# add ros
find_package(catkin REQUIRED COMPONENTS
roscpp
roslib
cmake_modules
)

set(THRID_PARTY_INCLUDE_DIRS ${THRID_PARTY_INCLUDE_DIRS} ${catkin_INCLUDE_DIRS})

catkin_package(
DEPENDS
CATKIN_DEPENDS roscpp roslib
INCLUDE_DIRS ${PROJECT_SOURCE_DIR}
LIBRARIES cppt_modules
)

# include
include_directories(
${THRID_PARTY_INCLUDE_DIRS}
${PROJECT_SOURCE_DIR}
)

add_library(cppt_modules SHARED
camera_model/camera.cpp
camera_model/pinhole_camera.cpp
camera_model/stereo_camera.cpp
)
target_link_libraries(cppt_modules ${THRID_PARTY_LIBS})

