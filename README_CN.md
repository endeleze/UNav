
# :rocket: [UNav](https://github.com/endeleze/UNav)

[English](README.md) **|** [简体中文](README_CN.md)**|** [แบบไทย](README_Thai.md)

---

UNav 是一个基于视觉的定位系统，它旨在帮助视障人士在不熟悉的环境中进行导航。

## :sparkles: 新功能

- 2023年5月29日： 支持 **Parallel RanSAC** 计算

<details>
  <summary>More</summary>

</details>

## :wrench: 安装 与 安装需要

- Python >= 3.8 (推荐使用 [Anaconda](https://www.anaconda.com/download/#linux) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.13](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

1. 克隆本 repository

    ```bash
    git clone https://github.com/endeleze/UNav.git
    ```

1. 安装所需的包
    APT packages
    ```bash
    sudo apt install < apt-requirements.txt
    ```
    Python modules
    ```bash
    pip install --upgrade pip setuptools wheel
    cd UNav
    pip install -r requirements.txt
    ```
## :computer: 使用
1. Server-Client 服务器与客户端

    * 设置 [server.yaml](configs/server.yaml) 与 [hloc.yaml](configs/hloc.yaml)， 按需求修改其中参数

   * 将数据按如下结构放入 **IO_root** 文件夹：
   
      ```bash
      UNav-IO/ 文件夹
      ├── data 文件夹
      │   ├── destination.json
      │   ├── PLACE 文件夹
      │   │   └── BUILDING 文件夹
      │   │       └── FLOOR 文件夹
      │   │           ├── access_graph.npy
      │   │           ├── boundaries.json
      │   │           ├── feats-superpoint.h5
      │   │           ├── global_features.h5
      │   │           ├── topo-map.json
      │   │           └── floorplan.png
      ```

      注意：如果没有 ***access_graph.npy*** 文件，你需要重新运行**step2_automatically.sh** 中的 [Path_finder_waypoints.py](./Path_finder_waypoints.py)
    * 运行服务器：
      ```bash
      source shell/server.sh
      ```
    * 运行客户端：
      * Jetson Board
      * Android
  
2. 可视化用户界面
    TODO

注意：UNav 只在 Ubuntu 进行测试，Windows 或 MacOS 并不适用。

## :hourglass_flowing_sand: TODO List

请看 [project boards](https://github.com/endeleze/UNav/projects).



## :e-mail: 联系

如果您有任何问题，请邮件联系 `ay1620@nyu.edu`.
