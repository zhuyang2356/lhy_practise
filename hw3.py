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
train_loader=DataLoader(train_set,batch_size=batch_size,
                        shuffle=True,num_workers=0,pin_memory=True)
valid_loader=DataLoader(valid_set,batch_size=batch_size,
                        shuffle=True,num_workers=0,pin_memory=True)
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
    # make sure the model is in eval mode,can turn off
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
        # max_prob=probs.argmax(dim=-1)
        max_prob=probs.topk(k=1,largest=True,dim=1)
        # max_prob>threshold
        # 返回probs矩阵里，大于0.09的数的index
        # (probs>0.09).nonzero()
        # max_prob是所有行，每行最大的数的定义，然后他们values大于阈值，.nonzero得到的是index矩阵，
        # 最后得到行index,说明这一行对应的img大于阈值
        (max_prob.values > 0.0945).nonzero()[:, 0]
        print(1)

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
do_semi=True

for epoch in range(n_epochs):
    # ---------- TODO ----------
    # In each epoch, relabel the unlabeled dataset for semi-supervised learning.
    # Then you can combine the labeled dataset and pseudo-labeled dataset for the training.
    if do_semi:
        # Obtain pseudo-labels for unlabeled data using trained model.
        pseudo_set = get_pseudo_labels(unlabeled_set, model)

        # Construct a new dataset and a data loader for training.
        # This is used in semi-supervised learning only.
        concat_dataset = ConcatDataset([train_set, pseudo_set])
        train_loader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    # Iterate the training set by batches.
    for batch in tqdm(train_loader):
        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()
        # Compute the gradients for parameters.
        loss.backward()
        # Clip the gradient norms for stable training.使用L2范数，对模型进行剪枝。
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)

    # The average loss and accuracy of the training set is the average of the recorded values.
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
          logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


