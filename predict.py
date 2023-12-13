import argparse
import os
import random
import sys

import pandas as pd
from torch.utils.data import DataLoader


from dataset.utils import read_predict_data, make_file
from dataset.dataloader import LoadPredictData
from model.model_cfg import *
from utils.trainer import *


def predict(args):
    name = make_file(args.save_path, args.name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers'.format(nw))

    # read model
    model = model_v2(num_classes=args.num_classes, d_model=32)
    model = model.to(device)

    model_weight_path = args.weight
    assert os.path.exists(model_weight_path), "file {} does not exist.".format(model_weight_path)
    model.load_state_dict(torch.load(model_weight_path, map_location='cpu'))

    # read data
    info, data = read_predict_data(args.root)

    predict_set = LoadPredictData(info.values, data.values, transform=None)

    predict_loader = DataLoader(predict_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                                collate_fn=LoadPredictData.collate_fn)

    predict_process_bar = tqdm(predict_loader, file=sys.stdout)

    model.eval()
    result = pd.DataFrame({"ID": [], "Time": [], "Risk Trade": []})
    with torch.no_grad():
        for _, item in enumerate(predict_process_bar):
            info, data = item
            outputs = model.forward(data.to(device))
            pred = torch.argmax(outputs, dim=1)
            predict_process_bar.desc = "Time : {} ID: {}  Risk Trade: {}.".format(info[0][1], info[0][0],
                                                                                  pred.cpu().numpy()[0])
            result=result.append(pd.DataFrame([[info[0][0], info[0][1], pred.cpu().numpy()[0]]], columns=result.columns))
    if args.save:
        result.to_csv(os.path.join(args.save_path, name, "result.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=r'C:\Users\lengxingcb\Desktop\GraduatioDesignExercise\train.csv',
                        help='data path')
    parser.add_argument("--batch_size", default=1)
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--model", default='v2')
    parser.add_argument("--weight", default='./save_weights/transformer0/best.pth')
    parser.add_argument("--save_path", default='./predict')
    parser.add_argument("--name", default='deeplearning_kfolder')
    parser.add_argument("--save", default=True)

    # parser.add_argument("--conf_threshold", default=0.25, help='conf threshold for classify')
    parser.add_argument("--num_workers", default=0)

    # random seed
    parser.add_argument("--random-seed", default=0)

    args = parser.parse_args()

    predict(args)
