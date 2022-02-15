#!conda env python
# -*- encoding: utf-8 -*-

import os
import shutil
from os.path import join as osp
import random

def create_txt(input,out_file):
    with open(out_file,'w') as f:
        if isinstance(input,list):
            f.writelines(str(line) + '\n' for line in input)
        else:
            for root,dirs,files in os.walk(input):
                for file in files:
                    f.write(file +'\n')


def copy_img(input_img,input_lbl,out_path):
    lbls=os.listdir(input_lbl)
    imgs = os.listdir(input_img)
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for img in imgs:
        if img in lbls:
            # print(img)
            shutil.copy2(osp(input_lbl,img),out_path+'/'+ img)


def move_img(input_dir,out_dir):
    for root,_, files in os.walk(input_dir):
            for file in files:
                print(file)
                shutil.move(os.path.join(root,file),os.path.join(out_dir,file))

def random_choose(input_dir,ratio):
    content=[]
    if os.path.isfile(input_dir):
        with open(input_dir,'r') as f:
            content = f.readlines()
    if os.path.isdir(input_dir):
        for root,_,files in os.walk(input_dir):
            for file in files:
                content.append(file)
            # content.append(file for file in files)
    # print(content)
    random.shuffle(content)
    if isinstance(ratio,int):
        return content[:ratio],content[ratio:]
    if isinstance(ratio,float):
        return content[:int(len(content)*ratio)],content[int(len(content)*ratio):]


