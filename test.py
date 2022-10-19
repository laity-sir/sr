import argparse
import time
import torch,os
import torch.backends.cudnn as cudnn
from data.utils import psnr,ssim,get_hr_lr_bicubic,preprocess_to_tesnor,preprocess_to_pil,preprocess_to_y,merge
import matplotlib.pyplot as plt
import config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, default='./network/best{}.pth'.format(config.scale))
    parser.add_argument('--image-file', type=str, default='./data/test/Set5/baby.png')
    parser.add_argument('--mode', type=str, help='y:表示ycbcr', default=config.mode)
    parser.add_argument('--scale', type=int, default=config.scale)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = torch.load(args.weights_file, map_location=lambda storage, loc: storage).to(device)
    model.eval()
    hr,lr,bicubic=get_hr_lr_bicubic(args.image_file,args.scale)  ###这里每个都是rgb通道
    basename = args.image_file.split('/')[-1]
    basename=basename.split('.')[0]   ###baby
    ##保存图片
    hr.save(os.path.join('./fig',basename+'__hr{}.png'.format(args.scale)))
    bicubic.save(os.path.join('./fig', basename + '__bicubic{}.png'.format(args.scale)))
    if args.mode=='y':
        print('在y通道进行测试')
        lr,_=preprocess_to_y(lr,device=device)
        hr1, ycbcr = preprocess_to_y(hr, device=device)
        bicubic1,_=preprocess_to_y(bicubic,device=device)
    else:
        print('在rgb进行测试')
        lr=preprocess_to_tesnor(lr,device)
        hr1=preprocess_to_tesnor(hr,device)
        bicubic1=preprocess_to_tesnor(bicubic,device)
    with torch.no_grad():
        end=time.time()
        preds = model(lr).clamp(0.0, 1.0)
        print('处理图片的时间',time.time()-end,'s')
    ##打印信息
    print('bicubic and hr psnr:{}'.format(psnr(hr1,bicubic1)))
    print('pred and hr psnr:{}'.format(psnr(hr1,preds)))
    print('bicubic and hr ssim:{}'.format(ssim(bicubic1,hr1)))
    print('pred and hr ssim:{}'.format(ssim(preds,hr1)))
    if args.mode=='y':
        output=merge(preds,ycbcr)
    else:
        output=preprocess_to_pil(preds)
    output.save(os.path.join('./fig', basename + '__pred{}.png'.format(args.scale)))
    ##显示
    plt.figure()
    plt.subplot(131)
    plt.imshow(hr)
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])
    plt.title("hr")
    plt.subplot(132)
    plt.imshow(bicubic)
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])
    plt.title("bicubic")
    plt.subplot(133)
    plt.imshow(output)
    plt.xticks([])  # 去掉x轴的刻度
    plt.yticks([])
    plt.title('pred')
    plt.show()