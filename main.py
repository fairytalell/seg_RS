#!conda env python
# -*- encoding: utf-8 -*-
# os.environ['CUDA_VISIBLE_DEVICES']='0,1'

import argparse
from tensorboardX import SummaryWriter
from networks.deeplab import DeepLab
from custom_dataset import CustomDataset
from torch.utils.data import DataLoader
from metrics import *
from utils import *
import random

# device =torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

setup_seed(10)


def train(opt):
    train_writer = SummaryWriter()

    #model
    model = DeepLab(num_classes=opt.num_classes, in_channels=opt.in_channels, pretrained=opt.pretrained, arch=opt.arch)
    # model.apply(weights_init_normal)
    if torch.cuda.is_available():
        model = nn.DataParallel(model.cuda(),device_ids=[opt.gpu_ids])


    #laod dataset
    train_dataset = CustomDataset(mode='trainR1',opt=opt)
    train_loader = DataLoader(train_dataset,opt.batchsize,shuffle=True,num_workers=opt.num_workers,pin_memory=True,drop_last=True)

    val_dataset = CustomDataset(mode='val',opt=opt)
    val_loader = DataLoader(val_dataset,opt.batchsize,num_workers=opt.num_workers,pin_memory=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=0).cuda()

    # lr
    lr = opt.lr
    optimizer = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=opt.weight_decay,betas=(0.9,0.9))

    # train
    model.train()
    best_OA = 0
    best_OA_epoch =0
    j=0
    for epoch in range(opt.max_epoch):
        epoch_loss =0
        j+=1
        for i,data in enumerate(train_loader):
            image ,target = data['image'],np.squeeze(data['label'])
            if torch.cuda.is_available():
                image,target = image.cuda(0,non_blocking=opt.non_blocking),target.cuda(0,non_blocking=opt.non_blocking)
            optimizer.zero_grad()
            output= model(image)
            loss = criterion(output,target.long())
            epoch_loss = epoch_loss + loss.item()
            train_writer.add_scalar('train loss',loss,j)
            optimizer.step()

        oa = val(model,val_loader)
        print('train loss: %.3f, val OA:%.3f' % (epoch_loss/len(train_loader),oa))
        if isinstance(model,torch.nn.DataParallel):
            model_dict = model.module.state_dict()
        else:
            model_dict=model.state_dict()
        if epoch == (opt.max_epoch -1):
            torch.save(model_dict,opt.savepath+'/model_last.pth')

        if oa > best_OA:
            best_OA_epoch = epoch
            best_OA = oa
            torch.save(model_dict,opt.savepath +'/model_bestOA.pth')

        print('\n------bestOA epoch:%d' % (best_OA_epoch))

    train_writer.close()


def val(model,dataloader):
    model.eval()
    with torch.no_grad():
        oa=0
        for i,data in enumerate(dataloader):
            image, target = data['image'], np.squeeze(data['label'])
            if torch.cuda.is_available():
                image= image.cuda()
            output = model(image)
            output = torch.argmax(output,1)
            predict = output.cpu().detach().numpy()
            target = target.cpu().numpy()
            oa += compute_global_accuracy(predict,target)
    oa /=(i+1)
    model.train()
    return oa

def main(args):
    for k,v in sorted(vars(args).items()):
        print(k,'=',v)
    train(args)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(description='Pytorch deeplabv3 for Segmentation Training')
    parse.add_argument('--dataset_dir', default='/media/ll/0D51AA6B1B7D664E/Dataset/DFC_Public_Dataset/4_seasons/visual_RGB/summer')
    parse.add_argument('--arch',default='resnet50',choices=['resnet50','resnet101'])
    parse.add_argument('--input_size',default=256)
    parse.add_argument('--gpu_ids',default=0,help='gpu id,eg. 0 0,1  0,1,2. use -1 for cpu')
    parse.add_argument('--dtype',default='RGB',help='dataset type: RGB')
    parse.add_argument('--lr',default=0.0001,type=float,help='learning rate')
    parse.add_argument('--savepath',default=None,help='checkpoint save path')
    parse.add_argument('--max_epoch',default=150,type=int)
    parse.add_argument('--in_channels',default=3)
    parse.add_argument('--pretrained',default=True)
    parse.add_argument('--batchsize',default=8)
    parse.add_argument('--num_workers',default=8)
    parse.add_argument('--weight_decay',default=1e-5)
    parse.add_argument('--non_blocking',default=True)

    opt = parse.parse_args()
    class_names, label_values = get_label_info(opt.dataset_dir+'/class_dict.txt')
    opt.num_classes = len(class_names)
    main(opt)






