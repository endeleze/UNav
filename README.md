
# :rocket: [UNav](https://github.com/endeleze/UNav)

[English](README.md) **|** [简体中文](README_CN.md)**|** [แบบไทย](README_Thai.md)

---

UNav is a vision-based location system designed to assist visually impaired individuals in navigating unfamiliar environments.

## :sparkles: New Features

- May 29, 2023. Support **Parallel RanSAC** computing 

<details>
  <summary>More</summary>

</details>

## :wrench: Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. Clone repo

    ```bash
    git clone https://github.com/endeleze/UNav.git
    ```

1. Create Python virtual environment
    ```bash
    cd UNav
    virtualenv env
    ```
    If you have not installed virtualenv, install it globally as follows.
    ```bash
    sudo pip3 install virtualenv
    ```

1. Activate the virtual environment
    ```bash
    source env/bin/activate
    ```

1. Install dependent packages
    * CUDA Toolkit
    ```bash
    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    chmod 755 cuda_12.1.1_530.30.02_linux.run
    ./cuda_12.1.1_530.30.02_linux.run
    ```
    * APT packages
    ```bash
    sudo apt install < apt-requirements.txt
    ```
    * From sources
    ** glog v0.4.0 (dependency for pyimplicite)
    ```bash
    git clone https://github.com/google/glog.git
    cd glog
    git checkout v0.4.0
    mkdir build
    cd build
    cmake ..
    make -j
    make install 
    ```
    ** Eigen v3.4.0 (dependency for pyimplicite)
    ```bash
    git clone https://gitlab.com/libeigen/eigen.git
    cd eigen
    git checkout 3.4.0
    mkdir build
    cd build
    cmake ..
    make install
    ```
    ** Ceres v2.1.0 (dependency for pyimplicite)
    ```bash
    git clone https://ceres-solver.googlesource.com/ceres-solver
    cd ceres-solver
    git checkout 2.1.0
    mkdir build
    cd build
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF
    make -j
    make install
    ldconfig
    ```
    ** GKlib (dependency for METIS)
    ```bash
    git clone https://github.com/KarypisLab/GKlib.git
    cd GKlib
    make config prefix=/usr/local
    make -j
    make install
    ```
    ** METIS (dependency for pyimplicite)
    ```bash
    git clone https://github.com/KarypisLab/METIS.git
    cd METIS
    git checkout v5.2.1
    make config cc=gcc prefix=/usr/local
    make install
    ```
    ** GMP (dependency for SuiteSparse)
    ```bash
    wget https://gmplib.org/download/gmp/gmp-6.2.1.tar.lz
    tar --lzip -xf gmp-6.2.1.tar.lz
    cd gmp-6.2.1
    ./configure
    make -j
    make check
    make install -j
    ```
    ** SuiteSparse (dependency for pyimplicite)
    ```bash
    git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
    cd SuiteSparse
    git checkout v6.0.0
    CMAKE_OPTIONS="-DENABLE_CUDA=1 -DCMAKE_CXX_FLAGS=-I/usr/local/cuda/include -DCMAKE_C_FLAGS=-I/usr/local/cuda/include" make global -j
    make install
    ```
    * Python modules
    ```bash
    pip3 install --upgrade pip setuptools wheel
    cd UNav
    pip3 install -r requirements.txt
    ```
## :computer: Using
1. Server-Client

    * Setup [server.yaml](configs/server.yaml) and tune [hloc.yaml](configs/hloc.yaml) us needed.

   * Put the data into **IO_root** you defined as following structure
   
      ```bash
      UNav-IO/
      ├── data
      │   ├── destination.json
      │   ├── PLACE
      │   │   └── BUILDING
      │   │       └── FLOOR
      │   │           ├── access_graph.npy
      │   │           ├── boundaries.json
      │   │           ├── feats-superpoint.h5
      │   │           ├── global_features.h5
      │   │           ├── topo-map.json
      │   │           └── floorplan.png
      ```

      Note that you need to rerun [Path_finder_waypoints.py](./Path_finder_waypoints.py) using **shell/step4.0.1.sh** if you do not have ***access_graph.npy***
    * Run server using
      ```bash
      source shell/server.sh
      ```
    * Run client device
      * Jetson Board
      * Android
  
2. Visualization-GUI
    TODO

Note that UNav is only tested in Ubuntu, and may be not suitable for Windows or MacOS.

## :hourglass_flowing_sand: TODO List

Please see [project boards](https://github.com/endeleze/UNav/projects).



## :e-mail: Contact

If you have any question, please email `ay1620@nyu.edu`.
