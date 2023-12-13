import argparse
import json
import os
import pickle

import joblib

from utils.preprocess import preprocess
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from dataset.utils import make_file


# RandomForest
def RFClassifier(cfg):
    with open(cfg, 'r') as file:
        rf_params = json.load(file)

    rfc = RandomForestClassifier(n_estimators=rf_params['n_estimators'],
                                 criterion=rf_params['criterion'],
                                 max_depth=rf_params['max_depth'],
                                 min_samples_split=rf_params['min_samples_split'],
                                 min_samples_leaf=rf_params['min_samples_leaf'],
                                 bootstrap=rf_params['bootstrap'],
                                 random_state=rf_params['random_state'])
    return rfc


# GradientBoosting
def GBClassifier(cfg):
    with open(cfg, 'r') as file:
        rf_params = json.load(file)

    gbc = GradientBoostingClassifier(n_estimators=rf_params['n_estimators'],
                                     criterion=rf_params['criterion'],
                                     max_depth=rf_params['max_depth'],
                                     min_samples_split=rf_params['min_samples_split'],
                                     min_samples_leaf=rf_params['min_samples_leaf'],
                                     random_state=rf_params['random_state'])

    return gbc


def train(args):
    print("Start training")
    name = make_file(args.save_path, args.name)

    features, labels = preprocess(path=args.data)
    rfc = RFClassifier(args.cfg)
    # 创建一个管道（Pipeline）实例，里面包含标准化方法和随机森林模型估计器
    pipeline = make_pipeline(StandardScaler(), rfc)
    # 设置交叉验证折数cv 表示使用带有十折的StratifiedKFold，再把管道和数据集传到交叉验证对象中

    scores = cross_val_score(pipeline, X=features, y=labels, cv=args.k, n_jobs=1)

    pipeline.fit(features, labels)
    print('Cross Validation accuracy scores: %s' % scores)
    print('Cross Validation accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))

    if args.save:
        # 保存模型到文件
        save_path = os.path.join(args.save_path, name, "random_forest_classify_model.pkl")

        with open(save_path, 'wb') as f:
            pickle.dump(pipeline, f)
    print("finish training")

def train_random_forest_classifier_with_grid_search(args):
    name = make_file(args.save_path, args.name)
    features, labels = preprocess(path=args.data)
    # 创建随机森林分类器
    rf_classifier = RandomForestClassifier()

    # 创建数据标准化处理流水线
    pipeline = make_pipeline(StandardScaler(), rf_classifier)

    # 定义网格搜索的参数网格
    print("GridSearch start")
    param_grid = {
        'randomforestclassifier__n_estimators': [50, 100],
        'randomforestclassifier__max_depth': [None, 5, 10]
    }

    # 创建GridSearchCV对象
    grid_search = GridSearchCV(pipeline, param_grid, cv=args.k)

    # 执行网格搜索和交叉验证
    grid_search.fit(features, labels)

    # 获取最佳模型
    best_model = grid_search.best_estimator_
    print(grid_search.cv_results_)
    # 保存训练好的模型

    if args.save_cfg:
        with open(os.path.join(args.save_path, name, "cfg_gridsearch.json"), 'w') as w:
            json.dump(grid_search.best_params_, w)
        # with open(os.path.join(args.save_path, name, "result_gridsearch.json"), "w") as w:
        #     json.dump(grid_search.cv_results_, w)

    if args.save:
        joblib.dump(best_model, os.path.join(args.save_path, name, "random_forest_model.pkl"))
    print("finish training")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data",
                        default=r"C:\Users\lengxingcb\Desktop\GraduatioDesignExercise\train.csv",
                        help='data.csv')

    parser.add_argument("--save", default=True, help="if save model")

    # parser.add_argument("--save_feature_importance", default=False, help="if save feature_importance to csv")

    # parser.add_argument("--GridSearch", default=False, help="GridSearch")
    # parser.add_argument("--save_cfg", default=True, help="if save GridSearch best model cfg")
    # parser.add_argument("--test_size", default=0.2, help="Training test set segmentation ratio ")
    parser.add_argument("--cfg", default="./rfc.json", help="random forest config file path")
    parser.add_argument("--k", default=3, help="k fold cross val")
    parser.add_argument("--save_path", default="./save_weights")
    parser.add_argument("--name", default="randomforestcls")

    # GridSearch cfg
    parser.add_argument("--GridSearch", default=False, help="GridSearch")
    parser.add_argument("--save_cfg", default=True, help="if save GridSearch best model cfg")

    args = parser.parse_args()

    if args.GridSearch:
        train_random_forest_classifier_with_grid_search(args)
    else:
        train(args)
