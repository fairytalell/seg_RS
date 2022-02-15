import os
import gdal
import numpy as np
import cv2
import tifffile

#哨兵2号R G B = 4 3 2
# uint16

#  读取tif数据集
def readTif(fileName):
    dataset = gdal.Open(fileName)
    if dataset == None:
        print(fileName + "文件无法打开")
    return dataset

#  保存tif文件函数
def writeTiff(im_data, im_geotrans, im_proj, path):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    elif len(im_data.shape) == 2:
        im_data = np.array([im_data])
        im_bands, im_height, im_width = im_data.shape
    # 创建文件
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path, int(im_width), int(im_height), int(im_bands), datatype)
    if (dataset != None):
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
    for i in range(im_bands):
        dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del dataset



def TifCrop(TifPath, SavePath, CropSize, RepetitionRate,aux_name):
    """
    :param TifPath:  影像路径
    :param SavePath:  裁剪后保存目录
    :param CropSize: 裁剪尺寸
    :param RepetitionRate: 重复率
    :return:
    """
    dataset_img = readTif(TifPath)
    width = dataset_img.RasterXSize

    height = dataset_img.RasterYSize
    proj = dataset_img.GetProjection()
    geotrans = list(dataset_img.GetGeoTransform())
    x_first = geotrans[0]
    y_first = geotrans[3]
    img = dataset_img.ReadAsArray(0, 0, width, height)  # 获取数据
    img=np.array(img,dtype=np.uint8)
    print('获取数据')

    #  获取当前文件夹的文件个数len,并以len+24命名即将裁剪得到的图像
    new_name = len(os.listdir(SavePath)) + 24
    #  裁剪图片,重复率为RepetitionRate

    '''
    img[C,H,W]
    '''
    for i in range(int((height - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
        for j in range(int((width - CropSize * RepetitionRate) / (CropSize * (1 - RepetitionRate)))):
            y_max = y_first + int(i * CropSize * (1 - RepetitionRate)) * geotrans[5]
            x_max = x_first + int(j * CropSize * (1 - RepetitionRate)) * geotrans[1]
            geotrans[0] = x_max
            geotrans[3] = y_max

            #  如果图像是单波段
            if (len(img.shape) == 2):
                cropped = img[
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            #  如果图像是多波段
            else:
                cropped = img[:,
                          int(i * CropSize * (1 - RepetitionRate)): int(i * CropSize * (1 - RepetitionRate)) + CropSize,
                          int(j * CropSize * (1 - RepetitionRate)): int(j * CropSize * (1 - RepetitionRate)) + CropSize]

            if np.mean(np.unique(cropped)) != 0.0:
                    #  写图像
                writeTiff(cropped, geotrans, proj, SavePath + '/'+ aux_name +'%d.tif' % (new_name))

                #  文件名 + 1
                new_name = new_name + 1
                print('写入第{}张图'.format(new_name))

