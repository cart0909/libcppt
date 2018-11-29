FROM nvidia/cuda:9.2-cudnn7-devel-ubuntu18.04
MAINTAINER Yi-Ren Liu

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install build-essential -y

RUN apt-get install tzdata -y\
    && apt-get install libatlas-base-dev liblapack-dev libblas-dev -y\
    && apt-get install libopenblas-dev libfftw3-dev liblapacke-dev -y\
    && apt-get install libsuitesparse-dev -y\
    && apt-get install libboost-all-dev -y\
    && apt-get install cmake cmake-qt-gui git gitk libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y\
    && apt-get install python-dev python-numpy python-pip libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev  libdc1394-22-dev -y\
    && apt-get install libgoogle-glog-dev libgtest-dev -y\
    && apt-get install zip -y\
    && apt-get install qtcreator -y\
    && apt-get install libyaml-cpp-dev -y\
    && apt-get install sudo wget gedit nano -y

WORKDIR /home/
RUN mkdir 3rdParty
WORKDIR 3rdParty/

RUN git clone https://github.com/jasperproject/jasper-client.git jasper \ 
    && chmod +x jasper/jasper.py \ 
    && pip install --upgrade setuptools \ 
    && pip install -r jasper/client/requirements.txt

WORKDIR /

# nvidia-docker2
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# install ROS
# install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release

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
RUN rosdep init \
    && rosdep update

# install ros packages
ENV ROS_DISTRO melodic
RUN apt-get update && apt-get install -y \
    ros-melodic-desktop-full

# install eigen (eigen 3.3.5) eigen-eigen-b3f3d4950030
WORKDIR /home/3rdParty
RUN wget http://bitbucket.org/eigen/eigen/get/3.3.5.zip && mv 3.3.5.zip eigen3.3.5.zip
RUN unzip eigen3.3.5.zip && mkdir -p eigen-eigen-b3f3d4950030/build
WORKDIR eigen-eigen-b3f3d4950030/build
RUN cmake .. && make install
WORKDIR /home/3rdParty
RUN rm -rf eigen-eigen-b3f3d4950030/

# install sophus
RUN git clone https://github.com/strasdat/Sophus && mkdir -p Sophus/build
WORKDIR Sophus
RUN git checkout b475c0a
WORKDIR build
RUN cmake .. && make -j12 && make install
WORKDIR /home/3rdParty
RUN rm -rf Sophus/build/

# install ceres
RUN git clone https://ceres-solver.googlesource.com/ceres-solver && mkdir -p ceres-solver/build
WORKDIR ceres-solver
RUN git checkout b040970
WORKDIR build
RUN cmake .. && make -j12 && make install
WORKDIR /home/3rdParty
RUN rm -rf ceres-solver/build/

# install opencv 3.4.4
RUN wget https://github.com/opencv/opencv/archive/3.4.4.zip
RUN wget https://github.com/opencv/opencv_contrib/archive/3.4.4.zip
RUN unzip opencv-3.4.4.zip && unzip opencv_contrib-3.4.4.zip && mkdir -p opencv-3.4.4/build
WORKDIR opencv-3.4.4/build
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=/home/3rdParty/opencv_contrib-3.4.4/modules -DWITH_CUDA=ON && make -j12 && make install
WORKDIR /home/3rdParty
RUN rm -rf opencv-3.4.4/ && rm -rf opencv_contrib-3.4.4/

# change ros cv_bridge opencv dependency
# Use "bash" as replacement for	"sh"
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
RUN wget https://github.com/ros-perception/vision_opencv/archive/2.1.0.zip
RUN unzip vision_opencv-2.1.0.zip && mkdir -p vision_opencv-2.1.0/cv_bridge/build
WORKDIR vision_opencv-2.1.0/cv_bridge/build
RUN source /opt/ros/$ROS_DISTRO/setup.bash\
&& cmake .. -DCMAKE_INSTALL_PREFIX=/opt/ros/melodic/\
&& make -j12\
&& make install
WORKDIR /home/3rdParty
RUN rm -rf vision_opencv-2.1.0/

# install mynt driver
RUN git clone https://github.com/slightech/MYNT-EYE-SDK-2
WORKDIR MYNT-EYE-SDK-2
RUN make init && make build && make install && make samples
RUN mkdir -p wrappers/ros/src/mynt_eye_ros_wrapper/build
WORKDIR wrappers/ros/src/mynt_eye_ros_wrapper/build
RUN source /opt/ros/melodic/setup.bash && cmake .. -DCMAKE_INSTALL_PREFIX="/opt/ros/melodic/"\
&& make -j8 && make install && mv /opt/ros/melodic/include/mynt_eye_ros_wrapper/launch /opt/ros/melodic/share/mynt_eye_ros_wrapper

# download arrayfire (must install by manual)
WORKDIR /usr/local
RUN wget http://arrayfire.s3.amazonaws.com/3.6.1/ArrayFire-v3.6.1_Linux_x86_64.sh && chmod +x ArrayFire-v3.6.1_Linux_x86_64.sh

# install evo
RUN pip install evo --upgrade --no-binary evo

# install brisk
WORKDIR /home/3rdParty
RUN git clone https://github.com/gwli/brisk && mkdir -p brisk/build
WORKDIR brisk/build
RUN cmake -DCMAKE_BUILD_TYPE=Release .. && make -j12 && make install
WORKDIR /home/3rdParty
RUN rm -rf brisk/build/

# erase apt list
RUN rm -rf /var/lib/apt/lists/*

WORKDIR /

# setup entrypoint
COPY ./ros_entrypoint.sh /
RUN chmod +x ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
