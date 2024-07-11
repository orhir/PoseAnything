# A Graph-Based Approach for Category-Agnostic Pose Estimation [ECCV 2024]
<a href="https://orhir.github.io/pose-anything/"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2311.17891"><img src="https://img.shields.io/badge/arXiv-2311.17891-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/orhir/PoseAnything)
[![Open in OpenXLab](https://cdn-static.openxlab.org.cn/app-center/openxlab_app.svg)](https://openxlab.org.cn/apps/detail/orhir/Pose-Anything)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/pose-anything-a-graph-based-approach-for/2d-pose-estimation-on-mp-100)](https://paperswithcode.com/sota/2d-pose-estimation-on-mp-100?p=pose-anything-a-graph-based-approach-for)

By [Or Hirschorn](https://scholar.google.co.il/citations?user=GgFuT_QAAAAJ&hl=iw&oi=ao) and [Shai Avidan](https://scholar.google.co.il/citations?hl=iw&user=hpItE1QAAAAJ)

This repo is the official implementation of "[A Graph-Based Approach for Category-Agnostic Pose Estimation](https://arxiv.org/pdf/2311.17891.pdf)".

<p align="center">
<img src="Pose_Anything_Teaser.png" width="384">
</p>

## 🔔 News
- **`11 July 2024`** Our paper will be presented at **ECCV 2024**.
- **`10 July 2024`** Uploaded new annotations - fix a small bug of DeepFashion skeletons.
- **`2 Feburary 2024`** Uploaded new weights - smaller models with stronger performance.
- **`20 December 2023`** Demo is online on [Huggingface](https://huggingface.co/spaces/orhir/PoseAnything) and [OpenXLab](https://openxlab.org.cn/apps/detail/orhir/Pose-Anything).
- **`7 December 2023`** Official code release.

## Introduction

We present a novel approach to CAPE that leverages the inherent geometrical relations between keypoints through a newly designed Graph Transformer Decoder. By capturing and incorporating this crucial structural information, our method enhances the accuracy of keypoint localization, marking a significant departure from conventional CAPE techniques that treat keypoints as isolated entities.

## Citation
If you find this useful, please cite this work as follows:
```bibtex
@misc{hirschorn2023pose,
      title={Pose Anything: A Graph-Based Approach for Category-Agnostic Pose Estimation},
      author={Or Hirschorn and Shai Avidan},
      year={2023},
      eprint={2311.17891},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Getting Started

### Docker [Recommended]
We provide a docker image for easy use.
You can simply pull the docker image from docker hub, containing all the required libraries and packages:

```
docker pull orhir/pose_anything
docker run --name pose_anything -v {DATA_DIR}:/workspace/PoseAnything/PoseAnything/data/mp100 -it orhir/pose_anything /bin/bash
```
### Conda Environment
We train and evaluate our model on Python 3.8 and Pytorch 2.0.1 with CUDA 12.1. 

Please first install pytorch and torchvision following official documentation Pytorch. 
Then, follow [MMPose](https://mmpose.readthedocs.io/en/latest/installation.html) to install the following packages:
```
mmcv-full=1.6.2
mmpose=0.29.0
```
Having installed these packages, run:
```
python setup.py develop
```

## Demo on Custom Images
<i>TRY IT NOW ON:</i> <a href="https://huggingface.co/spaces/orhir/PoseAnything">HuggingFace</a> / <a href="https://openxlab.org.cn/apps/detail/orhir/Pose-Anything">OpenXLab</a>


We provide a demo code to test our code on custom images. 

### Gradio Demo
We first require to install gradio:
```
pip install gradio==3.44.0
```
Then, Download the [pretrained model](https://drive.google.com/file/d/1RT1Q8AMEa1kj6k9ZqrtWIKyuR4Jn4Pqc/view?usp=drive_link) and run:
```
python app.py --checkpoint [path_to_pretrained_ckpt]
```
### Terminal Demo
Download
the [pretrained model](https://drive.google.com/file/d/1RT1Q8AMEa1kj6k9ZqrtWIKyuR4Jn4Pqc/view?usp=drive_link)
and run:

```
python demo.py --support [path_to_support_image] --query [path_to_query_image] --config configs/demo_b.py --checkpoint [path_to_pretrained_ckpt]
```
***Note:*** The demo code supports any config with suitable checkpoint file. More pre-trained models can be found in the evaluation section.


## Updated MP-100 Dataset
Please follow the [official guide](https://github.com/luminxu/Pose-for-Everything/blob/main/mp100/README.md) to prepare the MP-100 dataset for training and evaluation, and organize the data structure properly.

We provide an updated annotation file, which includes skeleton definitions, in the following [link](https://drive.google.com/drive/folders/1uRyGB-P5Tc_6TmAZ6RnOi0SWjGq9b28T?usp=sharing).

**Please note:**

Current version of the MP-100 dataset includes some discrepancies and filenames errors:
1. Note that the mentioned DeepFasion dataset is actually DeepFashion2 dataset. The link in the official repo is wrong. Use this [repo](https://github.com/switchablenorms/DeepFashion2/tree/master) instead.
2. We provide a script to fix CarFusion filename errors, which can be run by:
```
python tools/fix_carfusion.py [path_to_CarFusion_dataset] [path_to_mp100_annotation]
```

## Training

### Backbone Options
To use pre-trained Swin-Transformer as used in our paper, we provide the weights, taken from this [repo](https://github.com/microsoft/Swin-Transformer/blob/main/MODELHUB.md), in the following [link](https://drive.google.com/drive/folders/1-q4mSxlNAUwDlevc3Hm5Ij0l_2OGkrcg?usp=sharing).
These should be placed in the `./pretrained` folder.

We also support DINO and ResNet backbones. To use them, you can easily change the config file to use the desired backbone.
This can be done by changing the `pretrained` field in the config file to `dinov2`, `dino` or `resnet` respectively (this will automatically load the pretrained weights from the official repo).

### Training
To train the model, run:
```
python train.py --config [path_to_config_file]  --work-dir [path_to_work_dir]
```

## Evaluation and Pretrained Models
You can download the pretrained checkpoints from following [link](https://drive.google.com/drive/folders/1RmrqzE3g0qYRD5xn54-aXEzrIkdYXpEW?usp=sharing).

Here we provide the evaluation results of our pretrained models on MP-100 dataset along with the config files and checkpoints:

### 1-Shot Models
| Setting |                                                                       split 1                                                                       |                                                                       split 2                                                                       |                                                                       split 3                                                                       |                                                                       split 4                                                                       |                                                                       split 5                                                                       |
|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Tiny   |                                                                        91.19                                                                        |                                                                        87.81                                                                        |                                                                        85.68                                                                        |                                                                        85.87                                                                        |                                                                        85.61                                                                        |
|         |   [link](https://drive.google.com/file/d/1GubmkVkqybs-eD4hiRkgBzkUVGE_rIFX/view?usp=drive_link) / [config](configs/1shots/graph_split1_config.py)   |   [link](https://drive.google.com/file/d/1EEekDF3xV_wJOVk7sCQWUA8ygUKzEm2l/view?usp=drive_link) / [config](configs/1shots/graph_split2_config.py)   |   [link](https://drive.google.com/file/d/1FuwpNBdPI3mfSovta2fDGKoqJynEXPZQ/view?usp=drive_link) / [config](configs/1shots/graph_split3_config.py)   |   [link](https://drive.google.com/file/d/1_SSqSANuZlbC0utzIfzvZihAW9clefcR/view?usp=drive_link) / [config](configs/1shots/graph_split4_config.py)   |   [link](https://drive.google.com/file/d/1nUHr07W5F55u-FKQEPFq_CECgWZOKKLF/view?usp=drive_link) / [config](configs/1shots/graph_split5_config.py)   |
|  Small  |                                                                        94.73                                                                        |                                                                        89.79                                                                        |                                                                        90.69                                                                        |                                                                        88.09                                                                        |                                                                        90.11                                                                        |
|         | [link](https://drive.google.com/file/d/1RT1Q8AMEa1kj6k9ZqrtWIKyuR4Jn4Pqc/view?usp=drive_link) / [config](configs/1shot-swin/graph_split1_config.py) | [link](https://drive.google.com/file/d/1BT5b8MlnkflcdhTFiBROIQR3HccLsPQd/view?usp=drive_link) / [config](configs/1shot-swin/graph_split2_config.py) | [link](https://drive.google.com/file/d/1Z64cw_1CSDGObabSAWKnMK0BA_bqDHxn/view?usp=drive_link) / [config](configs/1shot-swin/graph_split3_config.py) | [link](https://drive.google.com/file/d/1vf82S8LAjIzpuBcbEoDCa26cR8DqNriy/view?usp=drive_link) / [config](configs/1shot-swin/graph_split4_config.py) | [link](https://drive.google.com/file/d/14FNx0JNbkS2CvXQMiuMU_kMZKFGO2rDV/view?usp=drive_link) / [config](configs/1shot-swin/graph_split5_config.py) |

### 5-Shot Models
| Setting |                                                                       split 1                                                                       |                                                                       split 2                                                                       |                                                                       split 3                                                                       |                                                                       split 4                                                                       |                                                                       split 5                                                                       |
|:-------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------:|
|  Tiny   |                                                                        94.24                                                                        |                                                                        91.32                                                                        |                                                                        90.15                                                                        |                                                                        90.37                                                                        |                                                                        89.73                                                                        |
|         |   [link](https://drive.google.com/file/d/1PeMuwv5YwiF3UCE5oN01Qchu5K3BaQ9L/view?usp=drive_link) / [config](configs/5shots/graph_split1_config.py)   |   [link](https://drive.google.com/file/d/1enIapPU1D8lZOET7q_qEjnhC1HFy3jWK/view?usp=drive_link) / [config](configs/5shots/graph_split2_config.py)   |   [link](https://drive.google.com/file/d/1MTeZ9Ba-ucLuqX0KBoLbBD5PaEct7VUp/view?usp=drive_link) / [config](configs/5shots/graph_split3_config.py)   |   [link](https://drive.google.com/file/d/1U2N7DI2F0v7NTnPCEEAgx-WKeBZNAFoa/view?usp=drive_link) / [config](configs/5shots/graph_split4_config.py)   |   [link](https://drive.google.com/file/d/1wapJDgtBWtmz61JNY7ktsFyvckRKiR2C/view?usp=drive_link) / [config](configs/5shots/graph_split5_config.py)   |
|  Small  |                                                                        96.67                                                                        |                                                                        91.48                                                                        |                                                                        92.62                                                                        |                                                                        90.95                                                                        |                                                                        92.41                                                                        |
|         | [link](https://drive.google.com/file/d/1p5rnA0MhmndSKEbyXMk49QXvNE03QV2p/view?usp=drive_link) / [config](configs/5shot-swin/graph_split1_config.py) | [link](https://drive.google.com/file/d/1Q3KNyUW_Gp3JytYxUPhkvXFiDYF6Hv8w/view?usp=drive_link) / [config](configs/5shot-swin/graph_split2_config.py) | [link](https://drive.google.com/file/d/1gWgTk720fSdAf_ze1FkfXTW0t7k-69dV/view?usp=drive_link) / [config](configs/5shot-swin/graph_split3_config.py) | [link](https://drive.google.com/file/d/1LuaRQ8a6AUPrkr7l5j2W6Fe_QbgASkwY/view?usp=drive_link) / [config](configs/5shot-swin/graph_split4_config.py) | [link](https://drive.google.com/file/d/1z--MAOPCwMG_GQXru9h2EStbnIvtHv1L/view?usp=drive_link) / [config](configs/5shot-swin/graph_split5_config.py) |

### Evaluation
The evaluation on a single GPU will take approximately 30 min. 

To evaluate the pretrained model, run:
```
python test.py [path_to_config_file] [path_to_pretrained_ckpt]
```
## Acknowledgement

Our code is based on code from:
 - [MMPose](https://github.com/open-mmlab/mmpose)
 - [CapeFormer](https://github.com/flyinglynx/CapeFormer)


## License
This project is released under the Apache 2.0 license.
