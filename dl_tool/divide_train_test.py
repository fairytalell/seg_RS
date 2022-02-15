import os
import shutil
import random
from os.path import join as osp

def divide_data(images_path,lbl_path,out_image,out_lbl,ratio):
    images=os.listdir(images_path)
    print(type(ratio))
    random.shuffle(images)
    print(images)
    train=[]
    val=[]
    if isinstance(ratio,int):
        train = images[:ratio]
        val = images[ratio:]
    if type(ratio) == 'float':
        train = images[:int(ratio * len(images))]
        val = images[int(ratio * len(images)):]
    trainimage_save = os.path.join(out_image,'train')
    os.makedirs(trainimage_save,exist_ok=True)

    trainlbl_save = os.path.join(out_lbl, 'train')
    os.makedirs(trainlbl_save, exist_ok=True)
    for file in train:

        shutil.copy2(osp(images_path, file), osp(trainimage_save, file))
        shutil.copy2(os.path.join(lbl_path,file),osp(trainlbl_save,file))


    valimage_save = os.path.join(out_image, 'val')
    os.makedirs(valimage_save, exist_ok=True)

    valbl_save = os.path.join(out_lbl, 'val')
    os.makedirs(valbl_save, exist_ok=True)
    for v in val:

        shutil.copy2(osp(images_path, v), osp(valimage_save, v))
        shutil.copy2(osp(lbl_path, v), osp(valbl_save, v))






