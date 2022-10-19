import argparse
import os
import sys
sys.path.append('../')
sys.path.append('../model')
import h5py
import numpy as np
import PIL.Image as pil_image
from utils import  convert_rgb_to_y,random_crop,augment
from PIL import Image
import config

def train(args):
    print('尺度因子是',args.scale)
    if not os.path.exists("../train"):
        os.makedirs("../train")
    h5_file = h5py.File(args.output_path, 'w')
    lr_patches = []
    hr_patches = []
    count = 0
    if args.mode=='y':
        print('生成y通道数据')
    else:
        print('生成rgb数据')
    for image_path in sorted(os.listdir(args.images_dir)):
        hr = pil_image.open(os.path.join(args.images_dir,image_path)).convert('RGB')
        if args.mode=='y':
            hr=np.array(hr)
            hr = convert_rgb_to_y(hr).astype(np.uint8)
            hr=Image.fromarray(hr)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        for i in range(args.samble_num):
            hr1=random_crop(augment(hr),(args.patch_size,args.patch_size))
            lr1 = hr1.resize((hr1.width // args.scale, hr1.height // args.scale), resample=pil_image.BICUBIC)
            count+=1
            if args.save:
                if not os.path.exists("../train/{}/lr/".format(args.scale)):
                    os.makedirs("../train/{}/lr/".format(args.scale))
                if not os.path.exists("../train/{}/hr/".format(config.scale)):
                    os.makedirs("../train/{}/hr/".format(args.scale))
                hr1.save(os.path.join("../train/{}/hr/".format(args.scale), '{}.png'.format(count)))
                lr1.save(os.path.join("../train/{}/lr/".format(args.scale), '{}.png'.format(count)))
            hr1 = np.array(hr1)
            lr1 = np.array(lr1)
            lr_patches.append(lr1)
            hr_patches.append(hr1)
    lr_patches = np.array(lr_patches)
    hr_patches = np.array(hr_patches)
    h5_file.create_dataset('lr', data=lr_patches)
    h5_file.create_dataset('hr', data=hr_patches)
    h5_file.close()
    print('over,训练集数量{}'.format(count))

def eval(args):
    if not os.path.exists("../test"):
        os.makedirs("../test")
    h5_file = h5py.File(args.eval_output_path, 'w')
    lr_group = h5_file.create_group('lr')
    hr_group = h5_file.create_group('hr')
    count=0
    if args.mode=='y':
        print('生成y通道数据')
    else:
        print('生成rgb数据')
    for i, image_path in enumerate(sorted(os.listdir(args.eval_images_dir))):
        hr = pil_image.open(os.path.join(args.eval_images_dir,image_path)).convert('RGB')
        if args.mode=='y':  ###这里是生成y通道数据
            hr=np.array(hr)
            hr = convert_rgb_to_y(hr).astype(np.uint8)
            hr=Image.fromarray(hr)
        hr_width = (hr.width // args.scale) * args.scale
        hr_height = (hr.height // args.scale) * args.scale
        hr = hr.resize((hr_width, hr_height), resample=pil_image.BICUBIC)
        lr = hr.resize((hr.width // args.scale, hr_height // args.scale), resample=pil_image.BICUBIC)
        if args.save:
            if not os.path.exists("../test/{}/lr/".format(args.scale)):
                os.makedirs("../test/{}/lr/".format(args.scale))
            if not os.path.exists("../test/{}/hr/".format(config.scale)):
                os.makedirs("../test/{}/hr/".format(args.scale))
            hr.save(os.path.join("../test/{}/hr/".format(args.scale), '{}.png'.format(i)))
            lr.save(os.path.join("../test/{}/lr/".format(args.scale), '{}.png'.format(i)))
        hr = np.array(hr)
        lr = np.array(lr)
        count+=1
        lr_group.create_dataset(str(i), data=lr)
        hr_group.create_dataset(str(i), data=hr)
    h5_file.close()
    print('测试集数量：{}'.format(count))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images-dir', type=str, default='./train')
    parser.add_argument('--output-path', type=str, default='../train/div2k_{}.h5'.format(config.scale))
    parser.add_argument('--eval_images-dir', type=str, default='./test/Set5')
    parser.add_argument('--eval-output-path', type=str, default='../test/set5_{}.h5'.format(config.scale))
    parser.add_argument('--scale', type=int, default=config.scale)
    parser.add_argument('--samble-num', type=int, default=20,help='随机采样的个数')
    parser.add_argument('--patch-size', type=int, default=config.patchsize)
    parser.add_argument('--mode', type=str,help='y:表示ycbcr',default=config.mode)
    parser.add_argument('--save', type=bool, default=False, help='是否保存图片')
    args = parser.parse_args()
    train(args)
    eval(args)