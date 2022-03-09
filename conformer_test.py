import torch
from torch import nn

x=torch.randn(128,4,80)




# conformer convolution
layernorm1=nn.LayerNorm(normalized_shape=80)
x1=layernorm1(x)
print(x1.shape)#128 4 80
# length 128 batch_size 4 channel 80
pwconv1=nn.Conv1d(in_channels=80,out_channels=120,kernel_size=1)
x1_1=x1.permute(1,2,0)
print("x1_1  shape:{}.{}.{}".format(x1_1.shape[0],x1_1.shape[1],x1_1.shape[2]))
# x2 4,120,128
# batch 4 channels 120 len 128
x2=pwconv1(x1_1)
gelu1=nn.GELU()
x3=gelu1(x2)
dwconv1=nn.Conv1d(in_channels=120,out_channels=200,kernel_size=3)
# x4 4,200,126
# batch_size 4 , channel 200 len 126
x4=dwconv1(x3)
bn1=nn.BatchNorm1d(num_features=200)
# x5 4,200,126
# batch_size 4,channel 200, len 126
x5=bn1(x4)
swish1=nn.SiLU()
x6=swish1(x5)
pwconv2=nn.Conv1d(in_channels=200,out_channels=120,kernel_size=1)
# x7 4,120,126 batch*channel*len
x7=pwconv2(x6)
dpout1=nn.Dropout(0.5)
x8=dpout1(x7)
print(1)