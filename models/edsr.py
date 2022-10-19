"""
    date:       2021/4/20 4:42 下午
    written by: neonleexiang
    https://github.com/NeonLeexiang/EDSR/blob/master/EDSR_model/edsr_model_pytorch.py
"""
import torch
import torch.nn as nn
from common import ResBlock,Upsampler,MeanShift

class model(nn.Module):
    """
    作者认为增加网络的宽度更加高效。
    n_resblocks:表示残差块的数量
    n_feats=256
    res_scale:残差缩放，作者在发现网络太宽模型出现了数值不稳定的现象，所以在每一个残差块的最后加一个残差缩放层来稳定训练
    """
    def __init__(self,scale=2, n_channels=3, n_resblocks=32, n_feats=256, res_scale=0.1, rgb_range=1):
        super(model, self).__init__()

        self.n_channels = n_channels
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats
        self.scale = scale
        self.res_scale = res_scale
        self.rgb_range = rgb_range

        self.kernel_size = (3, 3)
        self.padding = (1, 1)
        self.act = nn.ReLU(True)

        self.sub_mean = MeanShift(self.rgb_range)
        self.add_mean = MeanShift(self.rgb_range, sign=1)

        net_head = [nn.Conv2d(self.n_channels, self.n_feats, kernel_size=self.kernel_size, padding=self.padding)]
        net_body = [
            ResBlock(
                n_feats=self.n_feats, kernel_size=self.kernel_size, padding=self.padding,
                act=self.act, res_scale=self.res_scale
            ) for _ in range(self.n_resblocks)
        ]
        net_body.append(nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_feats,kernel_size=self.kernel_size, padding=self.padding))
        net_tail = [
            Upsampler(self.scale, self.n_feats, act=False),
            nn.Conv2d(in_channels=self.n_feats, out_channels=self.n_channels,
                      kernel_size=self.kernel_size, padding=self.padding)
        ]

        self.net_head = nn.Sequential(*net_head)
        self.net_body = nn.Sequential(*net_body)
        self.net_tail = nn.Sequential(*net_tail)

    def forward(self, x):
        x = self.sub_mean(x)
        x = self.net_head(x)

        res = self.net_body(x)
        res = torch.add(x, res)

        x = self.net_tail(res)
        x = self.add_mean(x)

        return x
if __name__=='__main__':
    from torchinfo import summary
    model=model(scale=2,n_feats=64,n_resblocks=3)
    x=torch.rand(1,3,32,32)
    summary(model,x.shape)