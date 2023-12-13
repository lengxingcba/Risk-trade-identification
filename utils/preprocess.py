import pandas as pd


def preprocess(path):
    """

    :param path:csv file
    :return:
    """
    data = pd.read_csv(path)

    features = data.iloc[:, 2:-1]
    label = data.iloc[:, -1]

    return features, label


def preprocess_predict(path):
    """

    :param path:csv file
    :return:
    """
    data = pd.read_csv(path)

    features = data.iloc[:, 2:]
    ID = data.iloc[:, 0]
    Time = data.iloc[:, 1]
    info = data.iloc[:, :2]

    return info, features
