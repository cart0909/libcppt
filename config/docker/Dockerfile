FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
MAINTAINER Yi-Ren Liu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update

RUN apt-get install apt-utils -y
RUN apt-get install build-essential -y

RUN apt-get install tzdata -y

# install BLAS ...
RUN apt-get install libatlas-base-dev liblapack-dev libblas-dev libopenblas-dev libfftw3-dev liblapacke-dev libsuitesparse-dev -y

# install boost library
RUN apt-get install libboost-all-dev -y

# install others
RUN apt-get install cmake cmake-qt-gui git gitk libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev python-dev python3-dev python-numpy python3-numpy python-pip python3-pip libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev libgoogle-glog-dev libgtest-dev zip qtcreator libyaml-cpp-dev sudo wget gedit nano bash-completion libv4l-dev dirmngr gnupg2 lsb-release libeigen3-dev -y

#RUN pip install python-igraph --upgrade # 3rdparty of kalibr

WORKDIR /home/

RUN git clone https://github.com/jasperproject/jasper-client.git jasper \ 
    && chmod +x jasper/jasper.py \ 
    && pip install --upgrade setuptools \ 
    && pip install -r jasper/client/requirements.txt

## install ROS
# setup keys
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys 421C365BD9FF1F717815A3895523BAEEB01FA116

# setup sources.list
RUN echo "deb http://packages.ros.org/ros/ubuntu `lsb_release -sc` main" > /etc/apt/sources.list.d/ros-latest.list

# install bootstrap tools
RUN apt-get update && apt-get install --no-install-recommends -y \
    python-rosdep \
    python-rosinstall \
    python-vcstools

# setup environment
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# bootstrap rosdep
RUN rosdep init && rosdep update

# install ros packages
ENV ROS_DISTRO melodic
RUN apt-get update && apt-get install ros-melodic-desktop-full ros-melodic-geographic-msgs ros-melodic-teleop-twist-keyboard ros-melodic-velodyne-simulator -y

## update gazebo
RUN sh -c 'echo "deb http://packages.osrfoundation.org/gazebo/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/gazebo-stable.list'
RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys D2486D2DD83DB69272AFE98867170598AF249743
RUN apt-get update
RUN apt-get upgrade -y

## install protobuf 3.0.0 (for gazebo 9 + ros melodic)
WORKDIR /home/
COPY protobuf-3.0.0.zip /home/
RUN unzip protobuf-3.0.0.zip
WORKDIR protobuf-3.0.0
RUN ./autogen.sh && ./configure && make && make install
WORKDIR /home/
RUN rm -rf protobuf-3.0.0/

## install sophus
WORKDIR /home/
RUN git clone https://github.com/strasdat/Sophus && mkdir -p Sophus/build
WORKDIR Sophus/build
RUN cmake .. && make -j12 && make install
WORKDIR /home/
RUN rm -rf Sophus/build/

# install ceres
RUN git clone https://ceres-solver.googlesource.com/ceres-solver && mkdir -p ceres-solver/build
WORKDIR ceres-solver/build
RUN cmake .. -DCERES_THREADING_MODEL=OPENMP -DEIGENSPARSE=ON -DCXSPARSE=OFF -DSUITESPARSE=OFF && make -j12 && make install
WORKDIR /home/
RUN rm -rf ceres-solver/build/

WORKDIR /

# nvidia-docker2
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# setup entrypoint
COPY ./ros_entrypoint.sh /
RUN chmod +x ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]

# others...
RUN apt-get install qt4-default -y

# get gazebo officical models
RUN mkdir -p ~/.gazebo/models && hg clone https://bitbucket.org/osrf/gazebo_models ~/.gazebo/models

# turtlebot3 package
RUN apt-get install ros-melodic-turtlebot3 -y

# rotors package dependency
RUN apt-get install ros-melodic-octomap* ros-melodic-mav-comm -y
RUN apt-get install ros-melodic-joystick-drivers -y

# ros usb camera package
RUN apt-get install ros-melodic-usb-cam -y

# catkin
RUN apt-get install python-catkin-tools -y

#import Carl Qt setting
COPY QtProject /root/.config/QtProject

# erase apt list
RUN rm -rf /var/lib/apt/lists/*
