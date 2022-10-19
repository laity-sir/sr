#####数据集
scale=4
eval_file='./test/set5_{}.h5'.format(scale)  ####测试集
train_file='./train/div2k_{}.h5'.format(scale)  ###训练集
patchsize=int(48*scale)         ####生成训练集的图片大小

##学习率
lr=1e-4
##
batchsize=64

mode='y'
if mode=='y':
    in_channel=1
else:
    in_channel=3

num_epoch=400

train_print_frequency = 10
resume=False

