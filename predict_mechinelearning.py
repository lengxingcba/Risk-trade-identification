import argparse
import json
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
from utils.preprocess import preprocess_predict


def predict(args):
    model_path = args.model
    data_path = args.data
    info, features = preprocess_predict(data_path)
    # 加载模型
    # r = pd.DataFrame()
    with open(model_path, 'rb') as f:
        loaded_model = pickle.load(f)

        pred = loaded_model.predict(features)

    samples = pd.concat([info, pd.DataFrame({"predict": pred})],axis=1)

    if args.save:
        samples.to_csv("result_all.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        default=r"C:\Users\lengxingcb\Desktop\GraduatioDesignExercise\pred.csv",
                        help='data')

    parser.add_argument("--model", default="./save_weights/randomforestcls0/random_forest_classify_model.pkl", help="model path")
    parser.add_argument("--save", default=True, help="if save result")
    parser.add_argument("--save_path", default="./predict", help="if save result")
    parser.add_argument("--name", default="randomforest", help="if save result")


    args = parser.parse_args()
    predict(args)
