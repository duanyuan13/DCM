# Why does Stereo Triangulation Not Work in UAV Distance Estimation

## Abstract
UAV distance estimation plays an important role for path planning of swarm UAVs and collision avoidance. However, the lack of annotated data seriously hinder the related studies. In this paper, we build and present a UAVDE dataset for UAV distance estimation, in which distance between two UAVs is obtained by UWB sensors. During experiments, we surprisingly observe that the commonly used stereo triangulation can not stand for UAV scenes. The core reason is the position deviation issue of UAVs due to long shooting distance and camera vibration, which is common in UAV scenes. To tackle this issue, we propose a novel position correction module (PCM), which can directly predict the offset between the image positions and the actual ones of UAVs and perform calculation compensation in stereo triangulation. Besides, to further boost performance on hard samples, we propose a dynamic iterative correction mechanism, which is composed of multiple stacked PCMs and a gating mechanism to adaptively determine whether further correction is required according to the difficulty of data samples. Consequently, the position deviation issue can be effectively alleviated. We conduct extensive experiments on UAVDE, and our proposed method can achieve a 38.84\% performance improvement, which demonstrates its effectiveness and superiority. The code and dataset would be released.

## Results
| Method      |       Val      |                |      Test      |                |
|-------------|:--------------:|:--------------:|:--------------:|:--------------:|
|             |     Abs Rel    |     Sq Rel     |     Abs Rel    |     Sq Rel     |
| Baseline    |      0.490     |      6.716     |      0.494     |      6.818     |
| + PCM       |      0.165     |      2.036     |      0.158     |      2.921     |
| + PCM + DIC |    **0.116**   |    **0.687**   |    **0.106**   |    **0.436**   |

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