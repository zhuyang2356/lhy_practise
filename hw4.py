import pandas as pd
import os,json,torch,random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from hw4DataPre import get_dataloader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from lrSched import get_cosine_schedule_with_warmup
import tqdm


class Classifier(nn.Module):
    def __init__(self,d_model=80,n_spks=600,dropout=0.1):
        super().__init__()
        self.parent=nn.Linear(40,d_model)
#         TODO:
#       use conformer https://arxiv.org/abs/2005.08100
        self.encoder_layer=nn.TransformerEncoderLayer(
            d_model=d_model,dim_feedforward=256,nhead=2
        )
        self.pred_layer=nn.Sequential(
            nn.Linear(d_model,d_model),
            nn.ReLU(),
            nn.Linear(d_model,n_spks),
        )
    def forward(self,mels):
        """
        :param mels:(batch_size,length,40)
        :return: (batch_size,n_spks)
        """
    # out: (batch size, length, d_model)
        out=self.parent(mels)
    # out: (length, batch size, d_model)
        out=out.permute(1,0,2)
    # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
    # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
    # mean pooling
        stats = out.mean(dim=1)
    # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out


if __name__=="__main__":
    data_dir="./Dataset"
    save_path="model.ckpt"
    batch_size=16
    n_workers=0
    valid_steps=2000
    warmup_steps=1000
    save_steps=10000
    total_steps=70000

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("use {} now".format(device))
    train_loader,valid_loader,speaker_num=get_dataloader(data_dir=data_dir,
                                    batch_size=batch_size,n_workers=n_workers)
    train_iterator=iter(train_loader)
    print("finish load data")
    model=Classifier(n_spks=speaker_num).to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=AdamW(model.parameters(),lr=1e-3)
    # TODO 为什么需要学习率热身，以及怎么实现
    # 学习率热身
    scheduler=get_cosine_schedule_with_warmup(optimizer,warmup_steps,total_steps)
    print("finish create model")

    best_accuracy=-1.0
    best_state_dict=None
    # pbar=tqdm(total=valid_steps,ncols=0,desc="Train",unit="step")

    for step in range(total_steps):
        try:
            batch=next(train_iterator)
        except StopIteration:
            train_iterator=iter(train_loader)
            batch=next(train_iterator)

        mels,labels=batch
        mels=mels.to(device)
        labels=labels.to(device)
        outs=model(mels)
        loss=criterion(outs,labels)
        # get the speakerId with hight prob
        preds=outs.argmax(1)
        accuracy=torch.mean((preds==labels).float())
#
        batch_loss=loss.item()
        batch_accuracy=accuracy.item()

#         update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        print("step loss")










