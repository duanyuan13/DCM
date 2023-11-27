# Why does Stereo Triangulation Not Work in UAV Distance Estimation

This repository is the official implementation of "Why does Stereo Triangulation Not Work in UAV Distance Estimation". It is designed for efficient UAV target distance estimation.

[Paper](https://arxiv.org/abs/2306.08939) | [Project Page](https://duanyuan13.github.io/UAVDE.html) | [YouTube]() | [BibeTex](#citation)

### Demo of Base Position Correction Module

<img src="./gif/pcm.gif"/>

### Demo of Position Correction Module with Dynamic Iterative Correction

![](./gif/dic_pcm.gif)

## Abstract
Distance estimation plays an important role for path planning and collision avoidance of swarm UAVs. However, the lack of annotated data seriously hinders the related studies. In this work, we build and present a UAVDE dataset for UAV distance estimation, in which distance between two UAVs is obtained by UWB sensors. During experiments, we surprisingly observe that the stereo triangulation cannot stand for UAV scenes. The core reason is the position deviation issue due to long shooting distance and camera vibration, which is common in UAV scenes. To tackle this issue, we propose a novel position correction module, which can directly predict the offset between the observed positions and the actual ones and then perform compensation in stereo triangulation calculation. Besides, to further boost performance on hard samples, we propose a dynamic iterative correction mechanism, which is composed of multiple stacked PCMs and a gating mechanism to adaptively determine whether further correction is required according to the difficulty of data samples. We conduct extensive experiments on UAVDE, and our method can achieve a significant performance improvement over a strong baseline (by reducing the relative difference from **49.4%** to **9.8%**), which demonstrates its effectiveness and superiority.

## Results
| Method      |    Val    |           |   Test    |           |
|-------------|:---------:|:---------:|:---------:|:---------:|
|             |  Abs Rel  |  Sq Rel   |  Abs Rel  |  Sq Rel   |
| Baseline    |   0.490   |   6.716   |   0.494   |   6.818   |
| + PCM       |   0.148   |   1.014   |   0.121   |   0.620   |
| + PCM + DIC | **0.114** | **0.673** | **0.098** | **0.401** |



## Quick Start
### Note: all codes can be implement on Win/Ubuntu, But Ubuntu is recomended.
Step1. Install PCM from source.
```shell
conda create -n pcm python=3.8 -y
conda activate pcm
pip install -r requirements.txt
```
Step2. Run  results from detector.(put the inferenced results of object detector into `.\data\`)

Positional Correction Module(PCM) only:
```shell
python run.py config/base_config.yaml
```
or the dynamic model
```shell
python run.py config/dynamic_base_config.yaml
```

## Data Preparation
### UAVDE Dataset
The UAVDE Dataset will be released soon. The samples is store in ./data/samples/

1. prepare your annotations.
```
the sample annotation is store in ./data/samples.csv
```

2. Your directory tree should be look like this:
```
UAV_Dataset
├── annotations.csv # annotations with ground truth distance
├── inference  
│   └── detected_labels # We recommended you to store detector results here
│       └── samples.txt # bounding box in yolo formats
├── test_no_prefix.txt
└── test.npy
```

## Citation
```
@misc{zhuang2023does,
      title={Why does Stereo Triangulation Not Work in UAV Distance Estimation}, 
      author={Jiafan Zhuang and Duan Yuan and Rihong Yan and Xiangyu Dong and Yutao Zhou and Weixin Huang and Zhun Fan},
      year={2023},
      eprint={2306.08939},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```