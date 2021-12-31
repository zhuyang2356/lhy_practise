import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset,DataLoader,Subset
from torchvision.datasets import DatasetFolder
from tqdm.auto import tqdm

train_tfm=transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])
test_tfm=transforms.Compose([
    transforms.Resize([128,128]),
    transforms.ToTensor(),
])

batch_size=16
# construct datasets
# The argument "loader" tells how torchvision reads the data
train_set=DatasetFolder("food-11/training/labeled",loader=lambda x:
                        Image.open(x),extensions="jpg",
                        transform=train_tfm)
valid_set=DatasetFolder("food-11/validation",loader=lambda x:
                        Image.open(x),extensions="jpg",
                        transform=test_tfm)
unlabeled_set=DatasetFolder("food-11/training/unlabeled",loader=lambda x:
                            Image.open(x),extensions="jpg",
                            transform=train_tfm)
test_set=DatasetFolder("food-11/testing",loader=lambda x:
                       Image.open(x),extensions="jpg",
                       transform=test_tfm)

# construct data loaders
# pin_memory是用来给GPU指定内存，让数据重新加载速度更快，cpu电脑可忽略
train_loader=DataLoader(train_set,batch_size=batch_size,
                        shuffle=True,num_workers=0,pin_memory=False)
valid_loader=DataLoader(valid_set,batch_size=batch_size,
                        shuffle=True,num_workers=0,pin_memory=False)
test_loader=DataLoader(test_set,batch_size=batch_size,shuffle=True)

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # input image size:[3,128,128]
        self.cnn_layers=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(4,4,0),
        )
        self.fc_layers=nn.Sequential(
            nn.Linear(256*8*8,256),
            nn.ReLU(),
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,11)
        )
    def forward(self,x):
        x=self.cnn_layers(x)
        x=x.flatten(1)
        x=self.fc_layers(x)
        return x


def get_pseudo_labels(dataset,model,threshold=0.65):
    device="cuda" if torch.cuda.is_available() else "cpu"
    data_loader=DataLoader(dataset,batch_size=batch_size,shuffle=True)
    # make sure the model is in eval mode
    model.eval()
    # define softmax function
    softmax=nn.Softmax(dim=-1)
    pseudo_set=[]
#     Iterate over the dataset by batches
    for batch in tqdm(data_loader):
        img,_=batch
        with torch.no_grad():
            logits=model(img.to(device))
        probs=softmax(logits)
        #--------------
        # TODO
        # filter the data and construct a new dataset
        # max_prob是所有行，每行最大的数的定义，然后他们values大于阈值，.nonzero得到的是index矩阵，
        max_prob=probs.topk(k=1,largest=True,dim=1)
        ident_index=(max_prob.values > threshold).nonzero()[:,0]
        ident_cate=(max_prob.values > threshold).nonzero()[:,1]
        temp=[batch[0][ident_index],ident_cate]
        pseudo_set.append(temp)
        print(1)
    # turn off the eval mode
    model.train()
    return pseudo_set

device="cuda" if torch.cuda.is_available() else "cpu"

# Initialize a model, and put it on the device specified
# 把模型放在指定GPU上
model=Classifier().to(device)
model.device=device
# 用crossEntropy作为模型性能指标
criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-5)

n_epochs=10

# whether to do semi-supervised learning
do_semi=True
for epoch in range(n_epochs):
#     TODO
#     in each epoch, relabel the unlabeled dataset for semi-supervised learning
#       then you combined the labeled dataset and pseudo-labeled dataset for training.
    if do_semi:
        pseudo_set=get_pseudo_labels(dataset=unlabeled_set,model=model)
#         construct a new dataset
        concat_dataset=ConcatDataset([train_set,pseudo_set])
        train_loader=DataLoader(concat_dataset,batch_size=batch_size,
                                shuffle=True,num_workers=0,pin_memory=False)
#    ------------ Training ---------------
#  让模型处于训练状态
    model.train()
#     训练信息
    train_loss=[]
    train_accs=[]
    for batch in tqdm(train_loader):
        imgs,labels=batch
        # 前向传播（模型和数据要在同一个GPU上）
        logits=model(imgs.to(device))
        loss=criterion(logits,labels.to(device))
        # 前一步的梯度要清零
        optimizer.zero_grad()
        loss.backward()
#         clip the gradient norms for stable training
#         clip_grad_norm，给梯度加入l2正则,max_norm是范数
        grad_norm=nn.utils.clip_grad_norm_(model.parameters(),max_norm=10)
#         用计算好的梯度更新参数
        optimizer.step()
#         计算当前batch的准确率
        acc=(logits.argmax(dim=-1)==labels.to(device)).float().mean()
#         记录loss 和 accuracy
        train_loss.append(loss.item())
        train_accs.append(acc)
    train_loss=sum(train_loss)/len(train_loss)
    train_acc=sum(train_accs)/len(train_accs)
    print("train loss nb_epochs:{},train_loss:{},train_accuray:{}".format(epoch,train_loss,train_acc))
#     ------------Validation---------------
#     make sure the model is in eval mode so that some modules like dropout are disabled
#     and work normally
#     让模型处于推理状态，有一些模块比如dropout就不工作了
    model.eval()
#
    valid_loss=[]
    valid_accs=[]
    for batch in tqdm(valid_loader):
        imgs,labels=batch
#         验证的时候不需要计算梯度
        with torch.no_grad():
            logits=model(imgs.to(device))
#         仍然可以计算loss
        loss=criterion(logits,labels.to(device))
        acc=(logits.argmax(dim=-1)==labels.to(device)).float().mean()
        valid_loss.append(loss.item())
        valid_accs.append(acc)
    valid_loss=sum(valid_loss)/len(valid_loss)
    valid_acc=sum(valid_accs)/len(valid_accs)
    print("valid epochs:{},valid_loss:{},valid_acc:{}".format(epoch,valid_loss,valid_acc))






