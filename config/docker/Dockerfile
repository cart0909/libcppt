FROM osrf/ros:melodic-desktop-full
MAINTAINER Yi-Ren Liu

RUN apt-get update
RUN apt-get install apt-utils -y
RUN apt-get install build-essential -y

RUN apt-get install libatlas-base-dev liblapack-dev libblas-dev -y
RUN apt-get install libopenblas-dev libfftw3-dev liblapacke-dev -y

RUN apt-get install libsuitesparse-dev -y

RUN apt-get install libboost-all-dev -y

RUN apt-get install cmake git libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev libswscale-dev -y
RUN apt-get install python-dev python-numpy python-pip libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev  libdc1394-22-dev -y

RUN git clone https://github.com/jasperproject/jasper-client.git jasper \ 
    && chmod +x jasper/jasper.py \ 
    && pip install --upgrade setuptools \ 
    && pip install -r jasper/client/requirements.txt

RUN apt-get install gitk -y

RUN apt-get install libgoogle-glog-dev libgtest-dev -y

RUN apt-get install zip -y

WORKDIR /
ADD eigen3.3.5.zip /
RUN unzip eigen3.3.5.zip
WORKDIR eigen3.3.5/
RUN mkdir build
WORKDIR build/
RUN cmake ..
RUN make install

WORKDIR /
ADD opencv-3.4.3.zip /
ADD opencv_contrib-3.4.3.zip /
RUN unzip opencv-3.4.3.zip
RUN unzip opencv_contrib-3.4.3.zip
WORKDIR opencv-3.4.3/
RUN mkdir build
WORKDIR build/
RUN cmake .. -DCMAKE_BUILD_TYPE=Release -DOPENCV_EXTRA_MODULES_PATH=/opencv_contrib-3.4.3/modules
RUN make -j12
RUN make install

WORKDIR /
RUN git clone https://ceres-solver.googlesource.com/ceres-solver
WORKDIR ceres-solver/
RUN git checkout b040970
RUN mkdir build
WORKDIR build/
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j12
RUN make test
RUN make install

WORKDIR /
RUN git clone https://github.com/strasdat/Sophus
WORKDIR Sophus/
RUN git checkout b475c0a
RUN mkdir build
WORKDIR build/
RUN cmake .. -DCMAKE_BUILD_TYPE=Release
RUN make -j12
RUN make install

WORKDIR /
RUN rm eigen3.3.5.zip
RUN rm -r eigen3.3.5/
RUN rm opencv-3.4.3.zip
RUN rm -r opencv-3.4.3/
RUN rm opencv_contrib-3.4.3.zip
RUN rm -r opencv_contrib-3.4.3/
RUN rm -r ceres-solver/
RUN rm -r Sophus/

# nvidia-docker2
# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# erase jasper folder
RUN rm -r jasper
