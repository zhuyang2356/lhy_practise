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
unlabeled_set=DatasetFolder("food-11/trainging/unlabeled",loader=lambda x:
                            Image.open(x),extensions="jpg",
                            transform=train_tfm)
test_set=DatasetFolder("food-11/testing",loader=lambda x:
                       Image.open(x),extensions="jpg",
                       transform=test_tfm)

# construct data loaders
train_loader=DataLoader(train_set,batch_size=batch_size,
                        shuffle=True,num_workers=2,pin_memory=True)
valid_loader=DataLoader(valid_set,batch_size=batch_size,
                        shuffle=True,num_workers=2,pin_memory=True)
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
    def forward(self):
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
#     Iterate over the dataset by batches
    for batch in tqdm(data_loader):
        img,_=batch
        with torch.no_grad():
            logits=model(img.to(device))
        probs=softmax(logits)
        #TODO filter the data and construct a new dataset

    # turn off the eval mode
    model.train()
    return dataset

device="cuda" if torch.cuda.is_available() else "cpu"

model=Classifier().to(device)
model.device=device

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-5)

n_epochs=10

# whether to do semi-supervised learning


