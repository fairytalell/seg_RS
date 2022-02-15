#!conda env python
# -*- encoding: utf-8 -*-

from torch.utils.data import Dataset

from utils import *
from torchvision.transforms import transforms as T
from skimage import io
import albumentations as albm
class CustomDataset(Dataset):

    def __init__(self,mode,opt,ratio=1):
        self.opt = opt
        self.root = opt.dataset_dir
        self.ratio = ratio
        self.dtype = opt.dtype
        self.insize = opt.input_size
        class_names, label_values = get_label_info(opt.dataset_dir+'/class_dict.txt')
        print(label_values)

        self.label_values = label_values

        self.transform = albm.Compose([
        albm.HorizontalFlip(p=0.5),
        albm.VerticalFlip(p=0.5),
        albm.ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.2,rotate_limit=45,p=0.2),
        albm.OneOf([
            albm.IAAAdditiveGaussianNoise(),
            albm.GaussianBlur(),
        ],p=0.2),
        albm.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        print(mode)

        self.lbls = filelist_fromtxt(self.root + '/' + mode + '_lbl', self.root + '/' + mode + '_lbl.txt')
        self.imgs = {}
        self.imgs = filelist_fromtxt(self.root + '/%s_%s' % (mode, self.dtype),self.root + '/%s_%s.txt' % (mode, self.dtype))

        assert len(self.lbls) == len(self.imgs)
        print('%s set contains %d images, a total of  categories.' % (mode, len(self.lbls)))
        print('Actual number of samples used = %d' % int(len(self.lbls) * self.ratio))

    def __len__(self):
        return int(len(self.lbls) * self.ratio)


    def __getitem__(self, index):
        data_dict = dict()
        lbl = io.imread(self.lbls[index])
        image = io.imread(self.imgs[index])
        as_tensor = T.ToTensor()

        data_dict['name'] = os.path.basename(self.lbls[index])
        transformed=self.transform(image=image, mask=lbl)
        data_dict['image'], data_dict['label'] = as_tensor(transformed['image']),as_tensor(transformed['mask'])
        return data_dict
