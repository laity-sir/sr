import torch
import torch.nn as nn

"""
计算求数据的均值,方差.
"""

def compute_para(model):
    """
    求模型的FLOPs,参数量
    models:定义的模型
    input:模型的输入尺寸 。模型的FLOPs跟特征图的大小有关，所以需要知道input的大小
    如果只计算卷积层的参数
    total_para_nums = 0
    for n,m in models.named_modules():
        if isinstance(m,nn.Conv2d):
            total_para_nums += m.weight.data.numel()
            if m.bias is not None:
                total_para_nums += m.bias.data.numel()
    print('total parameters:',total_para_nums）
    """
    total_para_num=sum([v.numel() for k, v in model.state_dict().items()])
    return total_para_num

def print_info(model,input):
    """
    打印模型的信息
        models=nn.Sequential(
        nn.Conv2d(3,64,3,1,1),
        nn.Conv2d(64,64,3,1,1),
        nn.Conv2d(64,3,3,1,1)
    )
    x=torch.rand(1,3,96,96)
    print_info(models,x)
    """
    from torchstat import stat
    from torchinfo import summary
    if torch.is_tensor(input):
        if input.dim()>3:
            print('summary models info')
            summary(model,input.shape)
            input=input.squeeze(0)
        print('stat models info ')
        stat(model,input.shape)
    else:
        print('输入信息错误')

"""
大核注意力机制，mobilenet
"""

if __name__=='__main__':


    from torchinfo import summary
    from torchstat import stat
