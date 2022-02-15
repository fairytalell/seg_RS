# -*- coding:utf-8 -*-
import numpy as np
from osgeo import gdal
import os
import glob
import math
import argparse
from os.path import join as pjoin

## 将小图合并为大图

#获取影像的左上角和右下角坐标
def GetExtent(in_fn):
    ds=gdal.Open(in_fn)
    geotrans=list(ds.GetGeoTransform())
    xsize=ds.RasterXSize
    ysize=ds.RasterYSize
    min_x=geotrans[0]
    max_y=geotrans[3]
    max_x=geotrans[0]+xsize*geotrans[1]
    min_y=geotrans[3]+ysize*geotrans[5]
    return min_x,max_y,max_x,min_y

def mosaic_img(path,savepath,name):
    os.chdir(path)
    os.makedirs(savepath,exist_ok=True)
    in_files = glob.glob("*.tif")
    in_fn = in_files[0]
    # 获取待镶嵌栅格的最大最小的坐标值
    min_x, max_y, max_x, min_y = GetExtent(in_fn)
    for in_fn in in_files[1:]:
        minx, maxy, maxx, miny = GetExtent(in_fn)
        min_x = min(min_x, minx)
        min_y = min(min_y, miny)
        max_x = max(max_x, maxx)
        max_y = max(max_y, maxy)
    # 计算镶嵌后影像的行列号
    in_ds_first = gdal.Open(in_files[0])
    geotrans = list(in_ds_first.GetGeoTransform())
    width = geotrans[1]
    height = geotrans[5]

    columns = math.ceil((max_x - min_x) / width)
    rows = math.ceil((max_y - min_y) / (-height))
    in_band = in_ds_first.GetRasterBand(1)

    driver = gdal.GetDriverByName('GTiff')

    out_ds = driver.Create( savepath + '/'+ name+'.tif', columns, rows, 1, in_band.DataType)
    out_ds.SetProjection(in_ds_first.GetProjection())
    geotrans[0] = min_x
    geotrans[3] = max_y
    out_ds.SetGeoTransform(geotrans)

    # 定义仿射逆变换
    inv_geotrans = gdal.InvGeoTransform(geotrans)
    # 开始逐渐写入
    for in_fn in in_files:
        in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()
        # 仿射逆变换

        offset = gdal.ApplyGeoTransform(inv_geotrans, in_gt[0], in_gt[3])
        x, y = map(int, offset)

        print(x, y)
        trans = gdal.Transformer(in_ds, out_ds, [])  # in_ds是源栅格，out_ds是目标栅格
        success, xyz = trans.TransformPoint(False, 0, 0)  # 计算in_ds中左上角像元对应out_ds中的行列号
        x, y, z = map(int, xyz)

        # print(x,y,z)
        for i in range(1):
            out_band = out_ds.GetRasterBand(i + 1)
            band = in_ds.GetRasterBand(i + 1)
            out_band.WriteArray(band.ReadAsArray(), x, y)

    del in_ds, out_ds

if __name__ == '__main__':
    parse=argparse.ArgumentParser(description='mosaic multi-images to a complete big image')
    parse.add_argument('--path',type=str,default=None)
    parse.add_argument('--savepath',type=str,default=None)
    args = parse.parse_args()
    for root, dirs, _ in os.walk(args.path):
        name = root.split('/')[-1]
        print(name)
        for dir in dirs:
            savepath = os.path.join(args.savepath,name)
            dir_path = os.path.join(root,dir)
            mosaic_img(dir_path,savepath,dir)

    # for path in pathes:
    #     args.path = path
    #     name = os.path.basename(path)
    #     # args.savepath = os.path.dirname(path)
    #     mosaic_img(args.path,args.savepath,name)





