cmake_minimum_required(VERSION 2.8.11)
project(cppt_player)

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

# add ros
find_package(catkin REQUIRED COMPONENTS
roscpp
roslib
cmake_modules
std_msgs
visualization_msgs
nav_msgs
sensor_msgs
image_transport
cv_bridge
tf
cppt_modules
)
catkin_package()

include_directories(${catkin_INCLUDE_DIRS})
include_directories(src)

add_executable(cppt_player
player_node.cpp
)

target_link_libraries(cppt_player
${catkin_LIBRARIES}
)


