# Unsupervised RGB-D Point Cloud Registration for Scenes with Low Overlap and Photometric inconsistency

This repository represents the official implementation of the paper:

[Unsupervised RGB-D Point Cloud Registration for Scenes with Low Overlap and Illumination Variation]

### Instructions
This code has been tested on 
- Python 3.8, PyTorch 1.12.1, CUDA 11.3, GeForce RTX 2080/NVIDIA Quadro P6000



#### Requirements
To create a virtual environment and install the required dependences please run:
```shell
conda create --name IG-UPCR python=3.8
conda activate IG-UPCR

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
We provide the pre-trained model of IG-UPCR in [Google Cloud](https://drive.google.com/drive/folders/1V2ZfkVNG1EG4oEsewJCaNlBU8HQSDYhr?usp=drive_link).

### Acknowledgments
In this project we use (parts of) the official implementations of the followin works: 

- [URR](https://github.com/mbanani/unsupervisedRR) (Trainer and dataset)
- [PointMBF](https://github.com/phdymz/PointMBF) (Network)
- [ScanNet](https://github.com/ScanNet/ScanNet) (Make dataset)
- [3DMatch](https://github.com/andyzeng/3dmatch-toolbox) (Make dataset)

 We thank the respective authors for open sourcing their methods. 



