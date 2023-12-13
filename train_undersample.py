import argparse
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset.utils import split_train_val_data_undersampling, one_cycle, make_file
from dataset.dataloader import LoadDataAndLabel
from model.model_cfg import *
from utils.loss import QFocalLoss, FocalLoss
from utils.trainer import *
import warnings

warnings.filterwarnings("ignore")


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    name = make_file(args.save_path, args.name)
    # 负样本欠采样策略，使每次输入模型训练的正负样本均衡
    data_train, data_val = split_train_val_data_undersampling(args.root, rate=0.2)

    train_true = data_train[data_train.iloc[:, -1] == 1]
    train_false = data_train[data_train.iloc[:, -1] == 0]

    val_data, val_label = data_val.iloc[:, 2:-1].values, data_val.iloc[:, -1].values
    val_data_set = LoadDataAndLabel(data=val_data, label=val_label, num_classes=args.num_classes, transform=None)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))

    val_loader = DataLoader(dataset=val_data_set,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=0,
                            collate_fn=LoadDataAndLabel.collate_fn)

    model = model_v2(args.num_classes, d_model=32)
    model = model.to(device)
    for para in model.parameters():
        para.requires_grad = True

    # 定义损失函数和优化器
    lossfunction = nn.BCEWithLogitsLoss(reduction="none")

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.cos_lr:
        lf = one_cycle(1, args.lrf, args.epochs)  # cosine
    else:
        lf = lambda x: (1 - x / args.epochs) * (1.0 - args.lrf) + args.lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # focalloss
    if args.focalloss:
        lossfunction = FocalLoss(lossfunction, alpha=args.focalloss_alpha, gamma=args.focalloss_gamma)
    best_acc = 0.0

    # record
    result = pd.DataFrame(
        {"epoch": [], "loss": [], "f1 score": [], "recall": [], "precision": [], "acc": [], "learning rate": []})
    # train model
    for epoch in range(args.epochs):
        f1, recall, precision = 0.0, 0.0, 0.0
        acc = 0.0

        # loader在循环内，每次都从原data_train中采样tf*len(正样本)数量的负样本
        data_train = pd.concat([train_true, train_false.sample(n=len(train_true) * args.tf, random_state=42)])
        train_data, train_label = data_train.iloc[:, 2:-1].values, data_train.iloc[:, -1].values
        train_data_set = LoadDataAndLabel(data=train_data, label=train_label, num_classes=args.num_classes,
                                          transform=None)
        train_loader = DataLoader(dataset=train_data_set,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=0,
                                  collate_fn=LoadDataAndLabel.collate_fn)

        train_process_bar = tqdm(train_loader, file=sys.stdout)
        model.train()
        running_loss = 0.0
        for batch in train_process_bar:
            optimizer.zero_grad()
            data, label = batch
            outputs = model(data.to(device))
            loss = lossfunction(outputs, label.to(device).float())
            running_loss += loss.mean().item()

            loss.mean().backward()

            train_process_bar.desc = "train epoch[{}/{}] loss:{:.6f} learning rate :{}".format(epoch + 1,
                                                                                               args.epochs,
                                                                                               running_loss,
                                                                                               optimizer.param_groups[
                                                                                                   0][
                                                                                                   'lr'])
        optimizer.step()
        scheduler.step()

        if args.val and (epoch + 1) % args.val_every_epoch == 0:
            val_process_bar = tqdm(val_loader, file=sys.stdout)
            model.eval()
            num_sample = 0
            num_acc = 0.0
            batch = 0
            predictions = []
            true_labels = []
            with torch.no_grad():
                for item in val_process_bar:
                    batch += 1
                    data, label = item
                    outputs = model(data.to(device))

                    predict_y = torch.max(outputs, dim=1)[1]
                    true_y = torch.max(label, dim=1)[1]
                    predictions.extend(predict_y.tolist())
                    true_labels.extend(true_y.tolist())
                    num_acc += torch.eq(predict_y, true_y.to(device)).sum().item()
                    num_sample += outputs.shape[0]
                    acc = num_acc / num_sample
                    val_process_bar.desc = "epoch: {}/{} f1 score: {:.6f} recall: {:.6f} precision: {:.6f} acc: {:.6f}".format(
                        epoch + 1, args.epochs,
                        f1, recall, precision, acc)
                f1 = f1_score(true_labels, predictions)
                recall = recall_score(true_labels, predictions)
                precision = precision_score(true_labels, predictions)
                print("epoch: {}/{} f1 score: {:.6f} recall: {:.6f} precision: {:.6f} acc: {:.6f}".format(
                    epoch + 1, args.epochs,
                    f1, recall, precision, acc))
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), f=os.path.join(args.save_path, name, "best.pth"))
        result = result.append(
            pd.DataFrame([[epoch, running_loss, f1, recall, precision, acc,
                           optimizer.param_groups[0]['lr']]], columns=result.columns))
        result.to_csv(os.path.join(args.save_path, name, "result.csv"), index=False)

    # save model
    torch.save(model.state_dict(), f=os.path.join(args.save_path, name, "last.pth"))
    print("finish training, best_acc: {}".format(best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=r'C:\Users\lengxingcb\Desktop\GraduatioDesignExercise\train.csv',
                        help='data path')
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--save_path", default='./save_weights')
    parser.add_argument("--name", default='Undersampling_training')

    parser.add_argument("--num_classes", default=2)

    parser.add_argument("--num_work", default=0)
    parser.add_argument("--epochs", default=400)
    parser.add_argument("--val", default=True)
    parser.add_argument("--val_every_epoch", default=20)

    parser.add_argument("--tf", default=10, help="每次采样负样本是正样本的几倍")

    # optimizer hyp
    parser.add_argument("--focalloss", default=True)
    parser.add_argument("--focalloss_alpha", default=0.75)
    parser.add_argument("--focalloss_gamma", default=3.5)

    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--lrf", default=0.00001)
    parser.add_argument("--cos_lr", default=True)

    args = parser.parse_args()
    train(args)
