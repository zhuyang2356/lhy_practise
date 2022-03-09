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
from tqdm import tqdm

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
        self.conformer=nn.Sequential(

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

def model_fn(batch, model, criterion, device):
  """Forward a batch through the model."""

  mels, labels = batch
  mels = mels.to(device)
  labels = labels.to(device)

  outs = model(mels)

  loss = criterion(outs, labels)

  # Get the speaker id with highest probability.
  preds = outs.argmax(1)
  # Compute accuracy.
  accuracy = torch.mean((preds == labels).float())

  return loss, accuracy

def valid(dataloader, model, criterion, device):
  """Validate on validation set."""
  model.eval()
  running_loss = 0.0
  running_accuracy = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss, accuracy = model_fn(batch, model, criterion, device)
      running_loss += loss.item()
      running_accuracy += accuracy.item()

    pbar.update(dataloader.batch_size)
    pbar.set_postfix(
      loss=f"{running_loss / (i+1):.2f}",
      accuracy=f"{running_accuracy / (i+1):.2f}",
    )

  pbar.close()
  model.train()

  return running_accuracy / len(dataloader)


if __name__=="__main__":
    data_dir="./Dataset"
    save_path="model.ckpt"
    batch_size=4
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
    pbar=tqdm(total=valid_steps,ncols=0,desc="Train",unit="step")

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
        preds=outs.argmax(1)
        accuracy=torch.mean((preds==labels).float())
        batch_loss=loss.item()
        batch_accuracy=accuracy.item()
#         update model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
#         log
        pbar.update()
        pbar.set_postfix(
            loss=f"{batch_loss:.2f}",
            accuracy=f"{batch_accuracy:.2f}",
            step=step+1,
        )
#
        if (step + 1) % valid_steps == 0:
            pbar.close()
            valid_accuracy = valid(valid_loader, model, criterion, device)
          # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()
            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

            # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")
    pbar.close()







