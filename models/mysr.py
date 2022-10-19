import torch
import torch.nn as nn

class MSRB_Block(nn.Module):
    def __init__(self):
        super(MSRB_Block, self).__init__()

        channel = 64
        self.conv_3_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_3_2 = nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, kernel_size=3, stride=1, padding=1,
                                  bias=True)
        self.conv_5_1 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.conv_5_2 = nn.Conv2d(in_channels=channel * 2, out_channels=channel * 2, kernel_size=5, stride=1, padding=2,
                                  bias=True)
        self.confusion = nn.Conv2d(in_channels=channel * 4, out_channels=channel, kernel_size=1, stride=1, padding=0,
                                   bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity_data = x
        output_3_1 = self.relu(self.conv_3_1(x))
        output_5_1 = self.relu(self.conv_5_1(x))

        input_2 = torch.cat([output_3_1, output_5_1], 1)
        output_3_2 = self.relu(self.conv_3_2(input_2))
        output_5_2 = self.relu(self.conv_5_2(input_2))
        output = torch.cat([output_3_2, output_5_2], 1)
        output = self.confusion(output)
        output = torch.add(output, identity_data)
        return output
# --------------------------MSRN------------------------------- #
class MSRN(nn.Module):
    def __init__(self,scale,in_channels=1):
        super(MSRN, self).__init__()
        out_channels_MSRB = 64
        self.scale = scale
        self.num=8
        ##浅层特征提取
        self.conv_input = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)

        ###深层特征提取
        self.residual = self.make_layer(MSRB_Block,num=self.num)

        ###瓶颈层，用于接收所有层的输出，降维
        self.bottle = nn.Conv2d(in_channels=out_channels_MSRB * 8 + 64, out_channels=64, kernel_size=1, stride=1,
                                padding=0, bias=True)
        self.conv = nn.Conv2d(in_channels=64, out_channels=64 * self.scale * self.scale, kernel_size=3, stride=1, padding=1,
                              bias=True)
        self.convt = nn.PixelShuffle(self.scale)
        self.conv_output = nn.Conv2d(in_channels=64, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=True)
    def make_layer(self, block,num):
        layers = []
        for i in range(num):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv_input(x)
        all_feature=[]
        all_feature.append(out)
        for i in range(self.num):
            out=self.residual[i](out)
            all_feature.append(out)
        out = torch.cat(all_feature, 1)
        out = self.bottle(out)
        out = self.convt(self.conv(out))
        out = self.conv_output(out)
        return out
if __name__=='__main__':
    model=MSRN(scale=2,in_channels=1)
    x=torch.rand(1,48,48)
    from torchstat import stat
    stat(model,x.shape)
    x=torch.rand(1,1,48,48)
    from torchinfo import summary
    summary(model,x.shape)