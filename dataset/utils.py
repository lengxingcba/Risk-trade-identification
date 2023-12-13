import math
import os
import random
import pandas as pd
import numpy as np


def split_train_val_data(root: str, rate=0.2):
    """
    将数据集按输入比例分割为训练和测试集，其中分割比例为从每个类别中采样比例，而不是整个数据集
    :param root: 文件地址
    :param rate: 分割比例
    :return: train,val
    """
    random.seed(0)
    assert os.path.exists(root), 'file {} is not exists'.format(root)

    data = pd.read_csv(root)
    # print('Total data: {}'.format(len(data)))
    label_data_dict = {}

    # 遍历数据，根据label字段将数据分组存放到不同的dataframe中(由于数据中正负样本比例严重失调，随机采样可能导致训练集或测试集中正样本比例很少，造成正样本被淹没在负样本中
    for label in data['Label'].unique():
        label_data_dict[label] = data[data['Label'] == label]

    data_val = pd.DataFrame()
    data_train = pd.DataFrame()
    for label, data in label_data_dict.items():
        n_samples = int(len(data) * rate)
        if n_samples == 0:
            n_samples = 1
        sampled_data = data.sample(n=n_samples)
        data_val = pd.concat([data_val, sampled_data], axis=0)
        # sampled_data_dict[label] = sampled_data
        unsampled_data = data.drop(sampled_data.index)
        data_train = pd.concat([data_train, unsampled_data], axis=0)
        # unsampled_data_dict[label]=unsampled_data
    # data_train.to_csv('train.csv',index=False)
    # data_val.to_csv('val.csv',index=False)

    return data_train.iloc[:, 2:-1].values, data_train.iloc[:, -1].values, data_val.iloc[:, 2:-1].values, data_val.iloc[
                                                                                                          :, -1].values


def split_train_val_data_undersampling(root: str, rate=0.2):
    """
    将数据集按输入比例分割为训练和测试集，其中分割比例为从每个类别中采样比例，而不是整个数据集
    :param root: 文件地址
    :param rate: 分割比例
    :return: train,val
    """
    random.seed(0)
    assert os.path.exists(root), 'file {} is not exists'.format(root)

    data = pd.read_csv(root)
    # print('Total data: {}'.format(len(data)))
    label_data_dict = {}

    # 遍历数据，根据label字段将数据分组存放到不同的dataframe中(由于数据中正负样本比例严重失调，随机采样可能导致训练集或测试集中正样本比例很少，造成正样本被淹没在负样本中
    for label in data['Label'].unique():
        label_data_dict[label] = data[data['Label'] == label]

    data_val = pd.DataFrame()
    data_train = pd.DataFrame()
    for label, data in label_data_dict.items():
        n_samples = int(len(data) * rate)
        if n_samples == 0:
            n_samples = 1
        sampled_data = data.sample(n=n_samples)
        data_val = pd.concat([data_val, sampled_data], axis=0)
        unsampled_data = data.drop(sampled_data.index)
        data_train = pd.concat([data_train, unsampled_data], axis=0)

    return data_train, data_val
    # return data_train.iloc[:,2:-1].values,data_train.iloc[:,-1].values,data_val.iloc[:,2:-1].values,data_val.iloc[:,-1].values


def split_kfold_cross_validation(root):
    """

    :param root:data.csv
    :param k: k-fold
    :return: data_true,data_false
    """
    random.seed(0)
    assert os.path.exists(root), 'file {} is not exists'.format(root)

    data = pd.read_csv(root)
    # print('Total data: {}'.format(len(data)))
    label_data_dict = {}

    # 遍历数据，根据label字段将数据分组存放到不同的dataframe中(由于数据中正负样本比例严重失调，随机采样可能导致训练集或测试集中正样本比例很少，造成正样本被淹没在负样本中
    for label in data['Label'].unique():
        label_data_dict[label] = data[data['Label'] == label]

    return label_data_dict.values()


def get_mean_std(data):
    """
    读取数据集，计算每个特征的均值和标准差，返回为mean:[f1_mean,f2_mean,...],std:[f1_std,f2_std...]
    :param data: dataframe
    :return: mean std
    """
    mean, var = [], []
    for column in range(len(data.columns)):
        feature = data.iloc[:, column]
        mean.append(feature.values.mean())
        var.append(feature.values.std())

    return mean, var


def one_cycle(y1=0.0, y2=1.0, steps=100):
    # lambda function for sinusoidal ramp from y1 to y2 https://arxiv.org/pdf/1812.01187.pdf
    return lambda x: ((1 - math.cos(x * math.pi / steps)) / 2) * (y2 - y1) + y1


def read_predict_data(path):
    """
    :param path: data_path
    :return: info,data
    """
    data = pd.read_csv(path)
    return data.iloc[:, :2], data.iloc[:, 2:]


def make_file(folder_path, name):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    folders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f)) and f.startswith(name)]
    if not folders:
        os.mkdir(os.path.join(folder_path, name))
    elif len(folders) >= 1:
        num = 0
        while os.path.exists(os.path.join(folder_path, name + str(num))):
            num += 1
        name = name+str(num)
        os.mkdir(os.path.join(folder_path, name))
    return name




def read_val_data(path):
    """
    :param path: data_path
    :return: info,data
    """
    data = pd.read_csv(path)
    return data.iloc[:, 2:-1].values, data.iloc[:, -1].values
