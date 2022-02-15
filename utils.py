#!conda env python
# -*- encoding: utf-8 -*-

import sys

import torch
import torch.nn as nn
import numpy as np
import os
import csv


def tensor2image(tensor):
    image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
    if image.shape[0] == 1:
        image = np.tile(image, (3, 1, 1))
    return image.astype(np.uint8)


class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


def weights_init_normal(m):

    if isinstance(m, nn.Conv2d):
        torch.nn.init.normal(m.weight.data, 0.0, 0.02)
    elif isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant(m.bias.data, 0.0)


def filelist_fromtxt(floder_dir,txt_path, ifPath=True,ratio=1.0):
    f=open(txt_path,'r')
    sourceInLines=f.readlines()
    f.close()
    namelist=[]

    for line in sourceInLines:
        img_name = line.strip('\n')
        if ifPath:
            img_name = os.path.join(floder_dir, img_name)
        namelist.append(img_name)
    namelist=sorted(namelist)
    if ratio != 1.0:
        return namelist[:int(len(namelist) * ratio)]
    else:
        return namelist


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_label_info(file_path):
    """
    Retrieve the class names and label values for the selected dataset.
    Must be in CSV or Txt format!

    Args:
        file_path: The file path of the class dictionairy

    Returns:
        Two lists: one for the class names and the other for the label values
    """

    filename, exten = os.path.splitext(file_path)
    if not (exten == ".csv" or exten == ".txt"):
        return ValueError("File is not a CSV or TxT!")

    class_names, label_values = [], []
    with open(file_path, 'r') as csvfile:
        file_reader = csv.reader(csvfile, delimiter=',')
        header = next(file_reader)  # skip one line
        print(header)
        for row in file_reader:
            if row != []:
                class_names.append(row[0])
                label_values.append([int(row[1]),int(row[2]),int(row[3])])
    return class_names, label_values

