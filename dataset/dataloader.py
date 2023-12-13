import numpy as np
import torch
from torch.utils.data import Dataset


def get_mean_std(data):
    """
    读取数据集，计算每个特征的均值和标准差，返回为mean:[f1_mean,f2_mean,...],std:[f1_std,f2_std...]
    :param data: dataframe
    :return: mean std
    """
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    return means, stds


class LoadDataAndLabel(Dataset):
    def __init__(self, data, label, num_classes, transform=None):
        # 前两列为ID与T_time，先默认最终结果与ID和时间无关
        # 数据标准化
        mean, std = get_mean_std(data)
        data = (data - mean) / std
        self.data = data
        self.label = label
        self.transform = transform
        self.num_classes=num_classes
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        label_onehot = [0]*self.num_classes

        data, label = self.data[item], self.label[item]
        label_onehot[label] = 1
        data = torch.FloatTensor(data)
        data = data.unsqueeze(0)
        # 如果不是16的倍数则填充
        # if data.shape[0] % 16 != 0:
        #     pad_size = 16 - (data.shape[0] % 16)
        #     data = torch.nn.functional.pad(data, (pad_size,),value=0)
        if data.shape[1] % 16 != 0:
            pad_size = 16 - (data.shape[1] % 16)
            data = torch.nn.functional.pad(data, (0, pad_size), value=0)

        if self.transform is not None:
            data = self.transform(data)

        return data, label_onehot

    @staticmethod
    def collate_fn(batch):
        data, label = tuple(zip(*batch))
        data = torch.stack(data, dim=0)
        label = torch.as_tensor(label)

        return data, label


class LoadPredictData(Dataset):
    def __init__(self, info, data,transform:None):
        super().__init__()
        mean, std = get_mean_std(data)
        data = (data - mean) / std

        self.info = info
        self.data = data

        self.transform=transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):

        info, data = self.info[item], self.data[item]

        data = torch.FloatTensor(data)
        data = data.unsqueeze(0)
        # 如果不是16的倍数则填充
        # if data.shape[0] % 16 != 0:
        #     pad_size = 16 - (data.shape[0] % 16)
        #     data = torch.nn.functional.pad(data, (pad_size,),value=0)
        if data.shape[1] % 16 != 0:
            pad_size = 16 - (data.shape[1] % 16)
            data = torch.nn.functional.pad(data, (0, pad_size), value=0)

        if self.transform is not None:
            data = self.transform(data)

        return info, data


    @staticmethod
    def collate_fn(batch):
        info, data = tuple(zip(*batch))
        data = torch.stack(data, dim=0)

        return info, data
