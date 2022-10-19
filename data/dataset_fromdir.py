import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from utils import convert_rgb_to_y,is_image_file,plot_data_loader_image,seed_torch
import config
def train_lr_transform(crop_size,scale):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(crop_size//scale,interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

def train_hr_transform(crop_size):
    return transforms.Compose([
        # transforms.CenterCrop(crop_size),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ToTensor()
    ])

def test_hr_transform():
    return transforms.Compose([
        transforms.ToTensor()  ###hr.shape[2],hr.shape[1]
    ])
def test_lr_transform(w,h,scale):
    w=int(w/scale)
    h=int(h/scale)
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((h,w),interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor()
    ])

class dataset(Dataset):
    def __init__(self, path, scale,patch_size=96,mode=config.mode,train=True,num=8):
        """
        path:高分辨率图片的路径
        scale:尺度因子
        patch_size=图片的大小
        mode:如果mode='y',则表示输入图片是y通道，否则是rgb格式
        train:如果为True,则表示训练集。
        """
        super(dataset, self).__init__()
        self.mode=mode
        self.train=train
        self.num=num
        self.scale = scale
        self.patch_size=patch_size
        hr = sorted(os.listdir(path))
        self.hr_name = [os.path.join(path, x) for x in hr if is_image_file(x)]

    def __len__(self):
        if self.train:
            return len(self.hr_name)*self.num
        else:
            return len(self.hr_name)

    def __getitem__(self, index):
        index_ = index % len(self.hr_name)
        hr = self.hr_name[index_]
        if self.mode=='y':
            hr = Image.open(hr).convert('RGB')
            y=convert_rgb_to_y(np.array(hr)).astype(np.uint8)
            y=Image.fromarray(y)
            hr=y
        else:
            hr=Image.open(hr).convert('RGB')
        if True:
            hr_width = (hr.width // self.scale) * self.scale
            hr_height = (hr.height // self.scale) * self.scale
            hr = hr.resize((hr_width, hr_height), resample=Image.BICUBIC)
        ###########################################################
        if self.train:
            hr=train_hr_transform(crop_size=self.patch_size)(hr)
            lr=train_lr_transform(crop_size=self.patch_size,scale=self.scale)(hr)
        else:
            hr=test_hr_transform()(hr)
            _,h,w=hr.shape
            lr=test_lr_transform(w,h,self.scale)(hr)
        return lr,hr

if __name__=='__main__':
    seed_torch()
    dataset1=dataset(path='test/Set5', scale=2, patch_size=100, train=1, num=8)
    dataloader=torch.utils.data.DataLoader(dataset=dataset1,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=0
                                           )
    plot_data_loader_image(data_loader=dataloader)
