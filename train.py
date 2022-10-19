import argparse
import os
import config
import numpy as np
import torch
if torch.cuda.is_available():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))
    print('显卡{}'.format(str(np.argmax([int(x.split()[2]) for x in open('tmp', 'r').readlines()]))))
    os.system('rm tmp')
import sys
sys.path.append('models')
sys.path.append('data')
import copy,time
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from model import MSRN
from data.datasets import TrainDataset,CPUPrefetcher, CUDAPrefetcher,EvalDataset
from data.utils import AverageMeter,psnr,ssim,seed_torch,save_checkpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, default=config.train_file)
    parser.add_argument('--eval-file', type=str, default=config.eval_file)
    parser.add_argument('--outputs-dir', type=str, default='./network')
    parser.add_argument('--scale', type=int, default=config.scale)
    parser.add_argument('--clip-grad', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=config.lr)
    parser.add_argument("--step", type=int, default=20,help="Sets the learning rate to the initial LR decayed by momentum every n epochs")
    parser.add_argument('--batch-size', type=int, default=config.batchsize)  #
    parser.add_argument("--momentum", default=0.9, type=float, help="Momentum, Default: 0.9")
    parser.add_argument("--weight-decay", "--wd", default=1e-8, type=float, help="weight decay, Default: 1e-4")
    parser.add_argument('--num-epochs', type=int, default=config.num_epoch)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1218)
    parser.add_argument("--resume", default="./network/x{}/models.pth".format(config.scale), type=str, help="Path to checkpoint, Default=None")
    parser.add_argument("--pretrained", default="", type=str, help='path to pretrained models, Default=None')
    parser.add_argument("--start-epoch", default=0, type=int, help="Manual epoch number (useful on restarts)")
    parser.add_argument('--mode', type=str, default=config.mode)
    args = parser.parse_args()

    args.outputs_dir = os.path.join(args.outputs_dir, 'x{}'.format(args.scale))

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    seed_torch(seed=args.seed)
    if args.mode=='y':
        print('数据是y通道的')
    else:
        print('数据是rgb格式的')
    print('尺度因子是{}'.format(config.scale))
    model = MSRN(scale=args.scale,in_channels=config.in_channel)
    model.to(device)
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    save_data={
    }
    if config.resume:
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["models"].state_dict())
            optimizer.load_state_dict(checkpoint['optimizer'])
            save_data=torch.load('./fig/save_data.pth')
            print("===> loading checkpoint: {},start_epoch: {} ".format(args.resume,args.start_epoch))
        else:
            print("===> no checkpoint found at {}".format(args.resume))
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("===> load models {}".format(args.pretrained))
            weights = torch.load(args.pretrained)
            model.load_state_dict(weights['models'].state_dict())
        else:
            print("===> no models found at {}".format(args.pretrained))

    train_dataset = TrainDataset(args.train_file)
    lenth=1000
    train_dataset,_=torch.utils.data.random_split(train_dataset,[lenth,len(train_dataset)-lenth])
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True)
    if torch.cuda.is_available():
        train_prefetcher = CUDAPrefetcher(train_dataloader,device)
    else:
        train_prefetcher = CPUPrefetcher(train_dataloader)
    eval_dataset=EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    print('导入数据集成功，训练集数量{}、测试集数量{}'.format(len(train_dataset),len(eval_dataset)))

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    writer = SummaryWriter(os.path.join("network", "logs{}".format(args.scale)), flush_secs=80)
    writer.add_graph(model.to(device),input_to_model=torch.randn(1,config.in_channel,24,24).to(device))
    len_traindataloader=len(train_prefetcher)
    for epoch in range(args.start_epoch, args.num_epochs+1):
        # 学习率衰减
        lr = args.lr * (0.5 ** ((epoch + 1) // args.step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        model.train()
        train_loss = AverageMeter()
        test_loss = AverageMeter()
        batch_time=AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size), ncols=80) as t:
            t.set_description('epoch: {}/{}'.format(epoch, args.num_epochs))
            #初始化
            batch_index=0
            train_prefetcher.reset()
            batch_data=train_prefetcher.next()
            end=time.time()
            while batch_data is not None:

                inputs=batch_data['lr'].to(device)
                labels=batch_data['hr'].to(device)
                ##释放不用的变量
                del batch_data
                torch.cuda.empty_cache()
                preds = model(inputs)
                loss = criterion(preds, labels)
                train_loss.update(loss.item(), len(inputs))
                optimizer.zero_grad()
                loss.backward()
                batch_time.update(time.time() - end)
#               #梯度裁剪
                nn.utils.clip_grad.clip_grad_norm_(model.parameters(), args.clip_grad / lr)
                optimizer.step()
                t.set_postfix(trainloss='{:.6f}'.format(train_loss.avg))
                t.update(len(inputs))
                del inputs, labels,preds
                torch.cuda.empty_cache()
                if batch_index%config.train_print_frequency==0:
                    writer.add_scalar('Train/Loss',loss.item(),batch_index+epoch*len_traindataloader)
                    for name, param in model.named_parameters(): ##显示梯度分布图，数据分布图
                        writer.add_histogram(tag=name + '_grad', values=param.grad, global_step=batch_index+epoch*len_traindataloader)
                        writer.add_histogram(tag=name + '_data', values=param.data, global_step=batch_index+epoch*len_traindataloader)
                del loss
                torch.cuda.empty_cache()
                end=time.time()
                batch_data=train_prefetcher.next()
                batch_index+=1
        model.eval()
        test_psnr = AverageMeter()
        test_ssim = AverageMeter()
        for data in eval_dataloader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                preds = model(inputs).clamp(0.0, 1.0)
                loss = criterion(preds, labels)
            test_loss.update(loss.item(), len(inputs))
            test_psnr.update(psnr(preds, labels), len(inputs))
            test_ssim.update(ssim(preds, labels), len(inputs))
            ###释放不用的变量
            del inputs, labels, loss
            torch.cuda.empty_cache()
        writer.add_scalar('Eval/loss',test_loss.avg,epoch)
        writer.add_scalar('Eval/psnr',test_psnr.avg,epoch)
        writer.add_scalar('Eval/ssim',test_ssim.avg,epoch)
        print('eval psnr: {:.2f},eval ssim:{:.2f} eval loss :{:.2f}，batch_data：{:.2f}'.format(test_psnr.avg, test_ssim.avg, test_loss.avg,batch_time.avg))
        save_data[epoch]={'train_loss':train_loss.avg,'test_loss':test_loss.avg,'test_psnr':test_psnr.avg,'test_ssim':test_ssim.avg}
        torch.save(save_data,'./fig/save_data.pth')
        ###这里只保存最后一次的数据
        if test_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = test_psnr.avg
            best_weights = copy.deepcopy(model)
            print('best_epoch:',best_epoch)
            ###保存整个网络
            torch.save(best_weights, os.path.join('./network', 'best{}.pth'.format(args.scale)))  ###直接保存整个模型
        save_checkpoint(args.outputs_dir, model, epoch, train_loss.avg, test_loss.avg, optimizer, test_psnr.avg, test_ssim.avg,best_epoch)
    print('best_epoch',best_epoch)