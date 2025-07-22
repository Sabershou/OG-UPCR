# Unsupervised RGB-D Point Cloud Registration for Scenes with Low Overlap and Photometric inconsistency

This repository represents the official implementation of the paper:
[Unsupervised RGB-D Point Cloud Registration for Scenes with Low Overlap and Photometric inconsistency]
## Introduction
Abstract: Point cloud registration is a fundamental task in 3D vision, playing a crucial role in various fields. With the rapid advancement of RGB-D sensors, unsupervised point cloud registration methods based on RGB-D sequences have demonstrated excellent performance. However, existing methods struggle in scenes with low overlap and photometric inconsistency. Low overlap results in numerous correspondence outliers, while photometric inconsistency hinders the model's ability to extract discriminative features. To address these challenges, we first propose the Overlapping Constraint for Inliers Detection (OCID) module, which filters and optimizes the initial correspondence set using an overlapping constraint. This module robustly selects reliable correspondences within the overlapping region while maintaining a balance between accuracy and efficiency. Additionally, we introduce a novel scene representation, 3DGS, which integrates both geometric and texture information, making it particularly well-suited for RGB-D registration tasks. Building on this, we propose the Gaussian Rendering for Photometric Adaptation (GRPA) module, which refines the geometric transformation and enhances the model's adaptability to scenes with inconsistent photometric information. Extensive experiments on ScanNet and ScanNet1500 demonstrate that our method achieves state-of-the-art performance.
![](https://github.com/Sabershou/OG-UPCR/blob/main/overview.png)

### Instructions
This code has been tested on 
- Python 3.8, PyTorch 1.12.1, CUDA 11.3, GeForce RTX 2080/NVIDIA Quadro P6000

#### Requirements
To create a virtual environment and install the required dependences please run:
```shell
conda create --name OG-UPCR python=3.8
conda activate OG-UPCR

pip install -r requirements.txt
```

### Make dataset 
You need to download the RGB-D version of 3DMatch dataset, ScanNet dataset and ScanNet1500 dataset in advance.
Details can refer to [URR](https://github.com/mbanani/unsupervisedRR/blob/main/docs/datasets.md) and [LoFTR](https://github.com/zju3dv/LoFTR).

#### 3DMatch
```shell
python create_3dmatch_rgbd_dict.py --data_root 3dmatch_train.pkl train
python create_3dmatch_rgbd_dict.py --data_root 3dmatch_valid.pkl valid
python create_3dmatch_rgbd_dict.py --data_root  3dmatch_test.pkl test
```

#### ScanNet
```shell
python create_scannet_dict.py --data_root scannet_train.pkl train
python create_scannet_dict.py --data_root scannet_valid.pkl valid
python create_scannet_dict.py --data_root scannet_test.pkl test 
```

### Train on 3DMatch
```shell
python train.py --name RGBD_3DMatch  --RGBD_3D_ROOT 
```

### Train on ScanNet
```shell
python train.py --name ScanNet  --SCANNET_ROOT 
```

### Inference
```shell
python test.py --dataset ScanNet/ScanNet_1500 --checkpoint --SCANNET_ROOT
```

### Pretrained Model
We provide the pre-trained model of OG-UPCR in [Google Cloud](https://drive.google.com/drive/folders/1V2ZfkVNG1EG4oEsewJCaNlBU8HQSDYhr?usp=drive_link).

### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [URR](https://github.com/mbanani/unsupervisedRR) (Trainer and dataset)
- [PointMBF](https://github.com/phdymz/PointMBF) (Network)
- [ScanNet](https://github.com/ScanNet/ScanNet) (Make dataset)
- [3DMatch](https://github.com/andyzeng/3dmatch-toolbox) (Make dataset)

 We thank the respective authors for open sourcing their methods. 



