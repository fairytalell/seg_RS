import os
import numpy as np
from osgeo import gdal
import glob
import argparse
from os.path import join as osp

## 16位转换为８位

def read_multi_tiff(input_file,r=4,g=3,b=2,mode='multi'): #mode is ['multi',''non-multi]
    """
    read tif image. if multi-band images, RGB band is saved.
    :param input_file:输入影像
    :return:波段数据，仿射变换参数，投影信息、行数、列数、波段数
    """
    print(input_file)
    dataset = gdal.Open(input_file)
    rows = dataset.RasterYSize
    cols = dataset.RasterXSize

    geo = dataset.GetGeoTransform()
    print(geo)

    proj = dataset.GetProjection()

    counts = dataset.RasterCount  # 波段数

    print('band count:{}'.format(counts))
    if mode=='multi':
            array_data = np.zeros((3, rows, cols))
            print(array_data.shape)
            R = dataset.GetRasterBand(r)
            G = dataset.GetRasterBand(g)
            B = dataset.GetRasterBand(b)
            array_data[0, :, :] = R.ReadAsArray()
            array_data[1, :, :] = G.ReadAsArray()
            array_data[2, :, :] = B.ReadAsArray()
            return array_data, geo, proj, rows, cols, 3
    else:
        dataset = dataset.ReadAsArray()
        return dataset, geo, proj, rows, cols, counts


def max_min_single(origin_16):
    array_data, geo, proj, rows, cols, counts = read_multi_tiff(origin_16,r=4,g=3,b=2)
    bands_max_min = np.zeros((3,1,2))
    for i in range(counts):
        band_max= np.max(array_data[i, :, :])
        band_min= np.min(array_data[i, :, :])
        bands_max_min[i]= [band_min,band_max]
    return bands_max_min

def max_min_final(fileList):
    min_max = max_min_single(fileList[0])
    for file in tif_files[1:]:
        band_max_min = max_min_single(file)
        for i in range(3):
            for k in range(1):
                for j in range(2):
                    if min_max[i][k][j] < band_max_min[i][k][j]:
                        min_max[i][k][j] = band_max_min[i][k][j]
    return min_max


def uint16_8_1(origin_16,min_max,output_8):
    array_data, geo, proj, rows, cols, counts = read_multi_tiff(origin_16, r=1, g=2, b=3)
    compress_data = np.zeros((counts, rows, cols))

    for i in range(counts):
        band_min = min_max[i, 0, 0]
        band_max = min_max[i, 0, 1]

        cutmin, cutmax = cumulativehistogram(array_data[i, :, :], rows, cols, band_min, band_max)
        print('cutmin:{},cutmax:{}'.format(cutmin,cutmax))

        compress_scale = (cutmax - cutmin) / 255

        array_data[array_data[i,:,:]<cutmin]= cutmin
        array_data[array_data[i, :, :] > cutmax] = cutmax
        compress_data[i,:,:] = (array_data[i,:,:] - cutmin) / compress_scale

    write_tiff(output_8, compress_data, rows, cols, counts, geo, proj)



def uint16_8(input_big,input_small=None, output_8=None,mode='small'):
    if mode=='big':
        array_data, geo, proj, rows, cols, counts = read_multi_tiff(input_big,r=4,g=3,b=2)

        compress_data = np.zeros((counts, rows, cols))

        for i in range(counts):
            # band_max = 2000
            # band_min= 0
            # cutmin= 0
            # cutmax= 1500
            band_max = np.max(array_data[i, :, :])
            band_min = np.min(array_data[i, :, :])

            cutmin, cutmax = cumulativehistogram(array_data[i, :, :], rows, cols, band_min, band_max)
            print('cutmin:{},cutmax:{}'.format(cutmin,cutmax))

            compress_scale = (cutmax - cutmin) / 255
            array_data[i,:,:][array_data[i,:,:]<cutmin]= cutmin
            array_data[i,:,:][array_data[i, :, :] > cutmax] = cutmax
            compress_data[i,:,:] = (array_data[i,:,:] - cutmin) / compress_scale

        write_tiff(output_8, compress_data, rows, cols, counts, geo, proj)

    #以大图为基准，可视化小图
    elif mode=='small':
        array_data, geo, proj, rows, cols, counts = read_multi_tiff(input_big,mode='RGB')
        compress_data = np.zeros((counts, rows, cols))
        print(input_small)
        images = glob.glob(os.path.join(input_small,'*.tif'))
        print(images)

        for image in images:
            name = os.path.basename(image)

            img = gdal.Open(image).ReadAsArray()
            for i in range(counts):
                band_max=np.max(array_data[i,:,:])
                band_mim=np.min(array_data[i,:,:])
                cutmin,cutmax = cumulativehistogram(array_data[i,:,:],rows,cols,band_mim,band_max)
                compress_scale = (cutmax-cutmin)/255
                img[i,:,:][img[i,:,:]<cutmin] = cutmin
                img[i, :, :][img[i, :, :] > cutmax] = cutmax
                compress_data[i, :, :] = (img[i, :, :] - cutmin) / compress_scale
            write_tiff(output_8 +'/' + name, compress_data, rows, cols, counts, geo, proj)


def write_tiff(output_file, array_data, rows, cols, counts, geo, proj):
    Driver = gdal.GetDriverByName("Gtiff")
    dataset = Driver.Create(output_file, cols, rows, counts, gdal.GDT_Byte)

    dataset.SetGeoTransform(geo)
    dataset.SetProjection(proj)

    for i in range(counts):
        band = dataset.GetRasterBand(i + 1)
        band.WriteArray(array_data[i, :, :])



def cumulativehistogram(array_data, rows, cols, band_min, band_max):
    """
    累计直方图统计
    """
    # 逐波段统计最值

    gray_level = int(band_max - band_min + 1)
    gray_array = np.zeros(gray_level)

    counts = 0
    for row in range(rows):
        for col in range(cols):
            gray_array[int(array_data[row, col] - band_min)] += 1
            counts += 1

    count_percent2 = counts * 0.02
    count_percent98 = counts * 0.98

    cutmax = 0
    cutmin = 0

    for i in range(1, gray_level):
        gray_array[i] += gray_array[i - 1]
        if (gray_array[i] >= count_percent2 and gray_array[i - 1] <= count_percent2):
            cutmin = i + band_min

        if (gray_array[i] >= count_percent98 and gray_array[i - 1] <= count_percent98):
            cutmax = i + band_min

    return cutmin, cutmax

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description='multi-channel 16-bit tiff image to RGB 8-bit images')
    parse.add_argument('--input_16bit_path',default=None,type=str, help='16 bit TIFF image Directory Path')
    parse.add_argument('--input_small',default=None,type=str,help='samll multi-channel images')
    parse.add_argument('--output_RGB_path',default=None,type=str,help='save as RGB Directory Path')
    args = parse.parse_args()

    if os.path.isdir(args.input_16bit_path):
        tif_files = glob.glob(osp(args.input_16bit_path,'*.tif'))
    else:
        tif_files = [args.input_16bit_path]
    os.makedirs(args.output_RGB_path,exist_ok=True)
    print(tif_files)
    for file in tif_files:
        print(file)
        basename = os.path.basename(file)
        print(basename)
        uint16_8(file,input_small=args.input_small,output_8=args.output_RGB_path,mode='small')






