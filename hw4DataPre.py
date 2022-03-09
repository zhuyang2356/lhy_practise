import pandas as pd
import os,json,torch,random
from pathlib import Path
from torch.utils.data import Dataset,random_split,DataLoader
from torch.nn.utils.rnn import pad_sequence

class myDataset(Dataset):
    def __init__(self,data_dir,segment_len=128):
        self.data_dir=data_dir
        self.segment_len=segment_len
        # load mapping from speaker name to their corresponding id
        mapping_path=Path(data_dir)/"mapping.json"
        mapping=json.load(mapping_path.open())
        self.speaker2id=mapping["speaker2id"]
#       load metadata of training data
        metadata_path=Path(data_dir)/"metadata.json"
        metadata=json.load(open(metadata_path))["speakers"]
#         获得所有speaker的数量
        self.speaker_num=len(metadata.keys())
        self.data=[]
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"],self.speaker2id[speaker]])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        feat_path,speaker=self.data[index]
#         load preprocessed mel-spectrogram
        mel=torch.load(os.path.join(self.data_dir,feat_path))
        if len(mel)>self.segment_len:
#             randomly get the starting point of the segment
            start=random.randint(0,len(mel)-self.segment_len)
#           get a segment with "segment_len" frames
            mel=torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel=torch.FloatTensor(mel)
        speaker=torch.FloatTensor([speaker]).long()
        return mel,speaker
    def get_speaker_number(self):
        return self.speaker_num
def collate_batch(batch):
  # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
  # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
  # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()

def get_dataloader(data_dir, batch_size, n_workers):
  """Generate dataloader"""
  dataset = myDataset(data_dir)
  speaker_num = dataset.get_speaker_number()
  # Split dataset into training dataset and validation dataset
  trainlen = int(0.9 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)

  train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=collate_batch,
  )
  valid_loader = DataLoader(
    validset,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
  )

  return train_loader, valid_loader, speaker_num