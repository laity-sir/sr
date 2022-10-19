import argparse
import time

import torch
from torch.utils.data.dataloader import DataLoader
import sys
sys.path.append('models')
sys.path.append('data')
from data.dataset_fromdir import  dataset
from data.utils import AverageMeter, psnr,ssim
import os
import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval-file', type=str, default='./data/test')
    parser.add_argument('--weights-file', type=str,default='./network/best{}.pth'.format(config.scale))
    parser.add_argument('--scale', type=int, default=config.scale)
    parser.add_argument('--mode', type=str, help='y:表示ycbcr', default=config.mode)
    args = parser.parse_args()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_dir_name=os.listdir(args.eval_file)
    print(test_dir_name)
    # models = models(scale=args.scale).to(device)
    print('loaded models')
    model=torch.load(args.weights_file, map_location=lambda storage, loc: storage).to(device)
    if args.mode=='y':
        print('在y通道上进行测试')
    for i in range(len(test_dir_name)):
        print('正在测试的数据集',test_dir_name[i])

        eval_dataset = dataset(os.path.join(args.eval_file, test_dir_name[i]), scale=args.scale, mode=args.mode, train=False)
        eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
        model.eval()
        epoch_psnr = AverageMeter()
        epoch_ssim = AverageMeter()
        epoch_time=AverageMeter()
        end=time.time()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
            epoch_psnr.update(psnr(preds, labels), len(inputs))
            epoch_ssim.update(ssim(preds, labels),len(inputs))
        epoch_time.update(time.time()-end)
        print('eval-file:{}    eval psnr: {:.2f},eval ssim:{:2f}  epoch time:{:f}'.format(test_dir_name[i], epoch_psnr.avg, epoch_ssim.avg,epoch_time.avg))
