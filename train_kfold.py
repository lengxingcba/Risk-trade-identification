import argparse
import os
import random
import sys

import pandas as pd
import torch
from sklearn.metrics import f1_score, recall_score, precision_score
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset.utils import split_train_val_data, split_train_val_data_undersampling, one_cycle, \
    split_kfold_cross_validation, make_file
from dataset.dataloader import LoadDataAndLabel
from model.model_cfg import *
from utils.loss import QFocalLoss, FocalLoss
from utils.trainer import *


def train(args):
    name = make_file(args.save_path, args.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # train_data, train_label, val_data, val_label = split_train_val_data(args.root, rate=0.2)

    data_false, data_true = split_kfold_cross_validation(args.root)
    len_false, len_true = len(data_false), len(data_true)

    part_true = [i for i in range(0, len_true, len_true // args.k)]
    part_false = [i for i in range(0, len_false, len_false // args.k)]

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))

    # data_transform = {
    #     'train': transforms.Compose([transforms.ToTensor(),
    #                                  # transforms.Normalize(mean, std)
    #                                  ]),
    #
    #     'val': transforms.Compose([transforms.ToTensor(),
    #                                # transforms.Normalize(mean, std)
    #                                ])
    # }

    # for data in train_loader:
    #     feature, label = data
    #
    #     print(label, len(label))

    # model
    # model = nn.Sequential(Block(1, 8, 30, 1),
    #                       Block(8, 16, 30, 1),
    #                       Block(16, 8, 30, 1),
    #                       Decoder(240))
    # model = model_v1(args.num_classes)
    # model =MLP(32,2)
    model = model_v2(num_classes=args.num_classes, d_model=32)
    model = model.to(device)
    for para in model.parameters():
        para.requires_grad = True

    # 定义损失函数和优化器
    lossfunction = nn.BCEWithLogitsLoss(reduction="none")
    # lossfunction=nn.CrossEntropyLoss(reduction="none")
    # lossfunction=nn.MSELoss(reduction="none")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if args.cos_lr:
        lf = one_cycle(1, args.lrf, args.epochs)  # cosine
    else:
        lf = lambda x: (1 - x / args.epochs) * (1.0 - args.lrf) + args.lrf  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)  # plot_lr_scheduler(optimizer, scheduler, epochs)

    # focalloss
    if args.focalloss:
        lossfunction = FocalLoss(lossfunction, alpha=0.25)
    best_acc = 0.0

    # record
    result = pd.DataFrame(
        {"epoch": [], "loss": [], "f1 score": [], "recall": [], "precision": [], "acc": [], "learning rate": []})

    # train model
    for epoch in range(args.epochs):
        acc_mean = 0.0
        loss_mean = 0.0
        f1, recall, precision = 0.0, 0.0, 0.0
        model.train()
        for i in range(args.k):
            # k折交叉验证，每次输入的训练数据：正样本正常k折交叉，负样本每次训练采样个数为正样本的n倍，验证集数据按k折交叉验证正常获取
            if i + 1 == args.k:
                data_val = pd.concat([data_true.iloc[part_true[i]:, :], data_false.iloc[part_false[i]:, :]],
                                     axis=0)
                sample = data_false.drop(data_false.iloc[part_false[i]:, :].index).sample(
                    (len_true - part_true[i]) * 10)
                data_train = pd.concat([data_true.drop(data_true.iloc[part_true[i]:, :].index), sample], axis=0)
                # data_train = pd.concat([data_true.drop(data_true.iloc[part_true[i]:, :].index),data_false.drop(data_false.iloc[part_false[i]:,:].index)],
                #                        axis=0)
            else:
                data_val = pd.concat([data_true.iloc[part_true[i]:part_true[i + 1], :],
                                      data_false.iloc[part_false[i]:part_false[i + 1], :]], axis=0)
                sample = data_false.drop(data_false.iloc[part_false[i]:part_false[i + 1], :].index).sample(
                    (part_true[i + 1] - part_true[i]) * 10)
                data_train = pd.concat([data_true.drop(data_true.iloc[part_true[i]:part_true[i + 1], :].index), sample],
                                       axis=0)
                # data_train = pd.concat([data_true.drop(data_true.iloc[part_true[i]:part_true[i + 1], :].index),
                #                         data_false.drop(data_false.iloc[part_false[i]:part_false[i + 1], :].index)],
                #                        axis=0)

            train_data, train_label = data_train.iloc[:, 2:-1].values, data_train.iloc[:, -1].values
            train_data_set = LoadDataAndLabel(data=train_data, label=train_label, transform=None,
                                              num_classes=args.num_classes)
            train_loader = DataLoader(dataset=train_data_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=0,
                                      collate_fn=LoadDataAndLabel.collate_fn)

            train_process_bar = tqdm(train_loader, file=sys.stdout)

            running_loss = 0.0
            for batch in train_process_bar:
                optimizer.zero_grad()
                data, label = batch
                output = model(data.to(device))
                # print(pred_class.dtype)
                loss = lossfunction(output, label.to(device).float())
                running_loss += loss.mean().item()
                # train_process_bar.desc('epoch: {} loss: {:.3f}'.format(epoch,loss))

                loss.mean().backward()

                train_process_bar.desc = "train epoch[{}/{} part {}/{}] loss:{:.6f} learning rate :{}".format(epoch + 1,
                                                                                                              args.epochs,
                                                                                                              i + 1,
                                                                                                              args.k,
                                                                                                              running_loss,
                                                                                                              optimizer.param_groups[0]['lr'])
            loss_mean += running_loss

            if args.val and (epoch + 1) % args.val_every_epoch == 0:

                val_data, val_label = data_val.iloc[:, 2:-1].values, data_val.iloc[:, -1].values
                val_data_set = LoadDataAndLabel(data=val_data, label=val_label, transform=None,
                                                num_classes=args.num_classes)

                val_loader = DataLoader(dataset=val_data_set,
                                        batch_size=batch_size,
                                        shuffle=False,
                                        num_workers=0,
                                        collate_fn=LoadDataAndLabel.collate_fn)

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
                    acc_mean += acc
                    if acc > best_acc:
                        best_acc = acc

                        torch.save(model.state_dict(), f=os.path.join(args.save_path, name, "best.pth"))

        acc_mean = acc_mean/args.k
        loss_mean = loss_mean/args.k
        result = result.append(
            pd.DataFrame([[epoch, loss_mean, f1, recall, precision, acc_mean,
                           optimizer.param_groups[0]['lr']]], columns=result.columns))
        result.to_csv(os.path.join(args.save_path, name, "result.csv"), index=False)
        optimizer.step()
        scheduler.step()

    # save model and result
    torch.save(model.state_dict(), f=os.path.join(args.save_path, name, "last.pth"))
    print("finish training, best_acc: {}".format(best_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=r'C:\Users\lengxingcb\Desktop\GraduatioDesignExercise\train.csv',
                        help='data path')
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--save_path", default='./save_weights')
    parser.add_argument("--name", default='transformer')

    # parser.add_argument("--conf_threshold", default=0.25, help='conf threshold for classify')
    parser.add_argument("--num_work", default=0)
    parser.add_argument("--epochs", default=300)
    parser.add_argument("--val", default=True)
    parser.add_argument("--val_every_epoch", default=10)

    parser.add_argument("--k", default=3, help="k fold")

    # optimizer hyp
    parser.add_argument("--focalloss", default=True)
    parser.add_argument("--lr", default=0.0001)
    parser.add_argument("--lrf", default=0.00001)
    parser.add_argument("--cos_lr", default=True)

    # random seed
    parser.add_argument("--random-seed", default=0)

    args = parser.parse_args()

    train(args)
