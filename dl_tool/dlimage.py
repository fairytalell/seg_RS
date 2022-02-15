#!conda env python
# -*- encoding: utf-8 -*-

import glob
from os.path import join as osp
from PIL import Image
import os
import cv2

def bmp2jpg(input,savepaths):
    img_suffixs = ['*_A.bmp', '*_B.bmp', '*_OUT.bmp']
    for suffix in img_suffixs:
        bmps = glob.glob(osp(input, suffix))
        for bmp in bmps:
            img = Image.open(bmp)
            base_name = os.path.basename(bmp).split('.')[0].split('_')[-1]
            name = os.path.basename(bmp).split('.')[0]
            print(name)
            os.makedirs(savepaths + '/' + base_name, exist_ok=True)
            img.save(savepaths + '/' + base_name + '/' + name + '.jpg')

def tif2jpg(tif,savepath):
    if not savepath.exists():
        os.makedirs(savepath, exist_ok=True)
    for root, _, files in os.walk(tif):
        for file in files:
            imgs = cv2.imread(osp(root, file), -1)
            cv2.imwrite(savepath + '/' + file.split('.')[0] + '.jpg', imgs)
            print('.......')



