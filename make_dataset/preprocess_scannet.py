import argparse
import glob
import math
import numpy as np
import os
import open3d as o3d
import cv2
import re
import shutil
import tqdm

def read_txt(filepath):
    sents = []
    with open(filepath, 'r') as f:
        # open为打开文件，r为读取
        f = open(filepath, 'r')
        # 逐行读取文件内容
        lines = f.readlines()
    for line in lines:
        res = re.split('\t', line)
        for index in range(len(res)):
            res[index] = str.strip(res[index])
            sents.append(res[index])
        # for i in range(len(line)):
        #     if line[i] == '\t':
        #         continue
        #     else:
        #         sents.append(line[i])
    return sents


def list_folders(path):
    folders = []
    for cur in os.listdir(path):
        if os.path.isdir(os.path.join(path, cur)) and not cur.startswith('.'):
            folders.append(cur)
    return folders


def process(cfg, name):
    txt_path = os.path.join(cfg.scannet_root, name)
    sequence = read_txt(txt_path)
    for seq in sequence:
        source_path = os.path.join(cfg.dataset_root, seq)
        target_path = os.path.join(cfg.out_root, name[:-4], 'scans', seq)
        if os.path.exists(target_path):
            # print(target_path)
            continue
        shutil.copytree(source_path, target_path)


def run_dataset(cfg, name):
    print("  Start dataset {} ".format(name[:-4]))

    process(cfg, name)

    print("  Finished dataset {} ".format(name[:-4]))


def run(cfg):
    print("Start processing")

    if not os.path.exists(cfg.out_root):
        os.makedirs(cfg.out_root)
    names = os.listdir(cfg.scannet_root)
    for name in names:
        if name == 'scannetv1_test.txt':
            continue
        run_dataset(cfg, name)

    print("Finished processing")


# ---------------------------------------------------------------------------- #
# Arguments
# ---------------------------------------------------------------------------- #
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_root', default='/media/zju231/Elements/scannet/ScanNet-rgbd')
    parser.add_argument('--scannet_root', default='/data/SYJ/Pycharmprojects/PointMBF-main/make_dataset/scannet')
    parser.add_argument('--out_root', default='/media/zju231/Elements/scannet/ScanNetRGBD')

    return parser.parse_args()


if __name__ == '__main__':
    cfg = parse_args()
    run(cfg)