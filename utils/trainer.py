import sys

import torch
from tqdm import tqdm


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, epoches):
    train_process_bar = tqdm(loader, file=sys.stdout)
    model.train()
    running_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()
        data, label = batch
        output = model(data.to(device))
        # pred_class = torch.max(output, dim=1)[1]
        # print(pred_class.dtype)
        loss = criterion(output, label.to(device))
        running_loss += loss.item()
        # train_process_bar.desc('epoch: {} loss: {:.3f}'.format(epoch,loss))

        loss.backward()
        optimizer.step()

        train_process_bar.desc = "train epoch[{}/{}] loss:{:.6f} learning rate :{}".format(epoch + 1,
                                                                                           epoches,
                                                                                           running_loss,
                                                                                           optimizer.param_groups[
                                                                                               0][
                                                                                               'lr'])


# 测试模型
def evaluate(model, loader, device, epoch, epoches):
    val_process_bar = tqdm(loader, file=sys.stdout)
    model.eval()
    num_sample = 0
    num_acc = torch.zeros(1).to(device)
    with torch.no_grad():
        for batch in loader:
            data, label = batch
            output = model(data.to(device))
            pred_class = torch.max(output, dim=1)[1]
            label = torch.max(label, dim=1)[1]
            num_acc += torch.eq(pred_class, label.to(device)).sum().item()
            num_sample += output.shape[0]

            val_process_bar.desc = "epoch: {}/{} acc: {:.6f}".format(epoch, epoches,
                                                                     num_acc / num_sample)