import pandas as pd
import torch
from torch.utils.data.dataloader import Dataset
import soundfile as sf


class PrepASV19Dataset(Dataset):
    def __init__(self, protocol_file_path, data_path, data_type='time_frame'):
        self.train_protocol = pd.read_csv(protocol_file_path, sep=' ', header=None)
        self.data_path = data_path
        self.data_type = data_type

    def __len__(self):
        return self.train_protocol.shape[0]

    def __getitem__(self, index):
        data_file_path = self.data_path + self.train_protocol.iloc[index, 1]

        if self.data_type == 'time_frame':
            sample, _ = sf.read(data_file_path + '.flac')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

        if self.data_type == 'CQT' or self.data_type == 'CQCC' or self.data_type == 'LFCC':
            sample = torch.load(data_file_path + '.pt')
            sample = torch.tensor(sample, dtype=torch.float32)
            sample = torch.unsqueeze(sample, 0)
            label = self.train_protocol.iloc[index, 4]
            label = label_encode(label)
            sub_class = self.train_protocol.iloc[index, 3]
            sub_class = sub_class_encode_19(sub_class)
            return sample, label, sub_class

    def get_weights(self):
        label_info = self.train_protocol.iloc[:, 4]
        num_zero_class = (label_info == 'bonafide').sum()
        num_one_class = (label_info == 'spoof').sum()
        weights = torch.tensor([num_one_class, num_zero_class], dtype=torch.float32)
        weights = weights / (weights.sum())
        return weights


def label_encode(label):
    if label == 'bonafide':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'human':
        label = torch.tensor(0, dtype=torch.int64)
    else:
        label = torch.tensor(1, dtype=torch.int64)
    return label


def sub_class_encode_19(label):
    # if environmentID == 'aaa':
    # PA
    if label == '-':
        label = torch.tensor(0, dtype=torch.int64)
    elif label == 'AA':
        label = torch.tensor(1, dtype=torch.int64)
    elif label == 'AB':
        label = torch.tensor(2, dtype=torch.int64)
    elif label == 'AC':
        label = torch.tensor(3, dtype=torch.int64)
    elif label == 'BA':
        label = torch.tensor(4, dtype=torch.int64)
    elif label == 'BB':
        label = torch.tensor(5, dtype=torch.int64)
    elif label == 'BC':
        label = torch.tensor(6, dtype=torch.int64)
    elif label == 'CA':
        label = torch.tensor(7, dtype=torch.int64)
    elif label == 'CB':
        label = torch.tensor(8, dtype=torch.int64)
    elif label == 'CC':
        label = torch.tensor(9, dtype=torch.int64)
    # LA
    elif label == 'A01':
        label = torch.tensor(10, dtype=torch.int64)
    elif label == 'A02':
        label = torch.tensor(11, dtype=torch.int64)
    elif label == 'A03':
        label = torch.tensor(12, dtype=torch.int64)
    elif label == 'A04':
        label = torch.tensor(13, dtype=torch.int64)
    elif label == 'A05':
        label = torch.tensor(14, dtype=torch.int64)
    elif label == 'A06':
        label = torch.tensor(15, dtype=torch.int64)
    elif label == 'A07':
        label = torch.tensor(16, dtype=torch.int64)
    elif label == 'A08':
        label = torch.tensor(17, dtype=torch.int64)
    elif label == 'A09':
        label = torch.tensor(18, dtype=torch.int64)
    elif label == 'A10':
        label = torch.tensor(19, dtype=torch.int64)
    elif label == 'A11':
        label = torch.tensor(20, dtype=torch.int64)
    elif label == 'A12':
        label = torch.tensor(21, dtype=torch.int64)
    elif label == 'A13':
        label = torch.tensor(22, dtype=torch.int64)
    elif label == 'A14':
        label = torch.tensor(23, dtype=torch.int64)
    elif label == 'A15':
        label = torch.tensor(24, dtype=torch.int64)
    elif label == 'A16':
        label = torch.tensor(25, dtype=torch.int64)
    elif label == 'A17':
        label = torch.tensor(26, dtype=torch.int64)
    elif label == 'A18':
        label = torch.tensor(27, dtype=torch.int64)
    elif label == 'A19':
        label = torch.tensor(28, dtype=torch.int64)

    return label
