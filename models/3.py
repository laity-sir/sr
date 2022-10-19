from torch import nn
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'

class res_block(nn.Module):
    """
    瓶颈残差块
    """
    def __init__(self,in_channels=64,feat=96,out_channels=64):
        super(res_block,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channels,out_channels=feat,kernel_size=1)
        self.conv2=nn.Conv2d(in_channels=feat,out_channels=feat,kernel_size=3,padding=1)
        self.conv3=nn.Conv2d(in_channels=feat,out_channels=out_channels,kernel_size=1)
        self.conv1_1=nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1)
        self.relu=nn.ReLU(inplace=True)
    def forward(self,x):
        residual=self.conv1_1(x)
        out=self.relu(self.conv1(x))
        out=self.relu(self.conv2(out))
        out=self.conv3(out)+residual
        out=self.relu(out)
        return out
class CALayer(nn.Module):
    def __init__(self, in_channel=64, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(in_channel, in_channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channel // reduction, in_channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y
class dm_block(nn.Module):
    """
    多尺度残差块
    """
    def __init__(self,in_channels=64,feat=32,out_channels=64):
        super(dm_block, self).__init__()
        self.num=4
        self.conv1=nn.Conv2d(in_channels,out_channels=feat,kernel_size=1)
        self.conv1_1 = nn.Conv2d(in_channels=feat, out_channels=feat, kernel_size=1)
        self.conv1_3 = nn.Conv2d(in_channels=feat, out_channels=feat, kernel_size=3, padding=3 // 2)
        self.conv1_5 = nn.Conv2d(in_channels=feat, out_channels=feat, kernel_size=5, padding=5 // 2)
        self.conv1_7 = nn .Conv2d(in_channels=feat,out_channels=feat ,kernel_size=7,padding=7//2)
        self.conv3=nn.Conv2d(in_channels=feat*self.num,out_channels=out_channels,kernel_size=1)
        self.relu=nn.ReLU(inplace=True)
        self.ca=CALayer()
    def forward(self,x):
        out=self.relu(self.conv1(x))
        o1=self.relu(self.conv1_1(out))
        o2=self.relu(self.conv1_3(out))
        o3=self.relu(self.conv1_5(out))
        o4=self.relu(self.conv1_7(out))
        out=self.conv3(torch.cat([o1,o2,o3,o4],1))
        out=self.relu(out+x)
        out=self.ca(out)
        return out

class cnn(nn.Module):
    """
    定义模型结构。
    整个模型有三部分组成，浅层特征提取、深层特征映射、重建
    """
    def make_layer(self, block, num_of_layer):
        """
        构成级联的模块，block是模块的种类，num_of_layer 是模块的数量
        """
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)
    def __init__(self,scale,in_channels=1,out_channels=64):
        super(cnn, self).__init__()
        self.num=18
        self.residual=nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=1),
        )
        # self.SFE=msr_block(in_channels=in_channels,out_channels=out_channels)
        self.SFE = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        ###级联残差模块
        self.DM =self.make_layer(res_block,self.num)
        # self.DM=dms_block(in_channels=out_channels,feat=96,out_channels=out_channels)
        # self.DFE=msr_block(in_channels=out_channels,out_channels=out_channels)
        self.DFE=nn.Sequential(
            nn.Conv2d(in_channels=out_channels,out_channels=out_channels,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
        )
        self.aff=nn.Sequential(
            nn.Conv2d(in_channels=out_channels*(self.num+2),out_channels=out_channels,kernel_size=1),
            nn.ReLU(inplace=True)
        )
        self.up_sample=nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=in_channels * (scale ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale)
        )
        for para in self.modules():
            if isinstance(para,nn.Conv2d):
                nn.init.kaiming_normal_(para.weight)
                if para.bias is not None:
                    para.bias.data.zero_()
    def forward(self,x):
        residual=self.residual(x)
        out=self.SFE(x)
        all_feature=[]
        all_feature.append(out)
        for i in range(self.num):
            out=self.DM[i](out)
            all_feature.append(out)
        out=self.DFE(out)
        all_feature.append(out)
        out=self.aff(torch.cat(all_feature,1))+residual
        out=self.up_sample(out)
        return out

if __name__=='__main__':
    from torchinfo import summary
    model=cnn(2)
    print(model)
    x=torch.rand(1,1,48,48)
    output=model(x)
    print(output.shape)
    print(x.shape)
    summary(model,x.shape)
