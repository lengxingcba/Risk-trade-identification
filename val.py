import argparse
import json
import os
from sklearn.metrics import f1_score, recall_score, precision_score
import pandas as pd
from torch.utils.data import DataLoader

from dataset.utils import read_val_data, make_file
from dataset.dataloader import LoadDataAndLabel
from model.model_cfg import *
from utils.trainer import *

import warnings
warnings.filterwarnings("ignore")

def predict(args):
    name = make_file(args.save_path, args.name)
    num_acc = 0.0
    num_sample = 0

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
    data, label = read_val_data(args.root)

    val_set = LoadDataAndLabel(data, label, num_classes=args.num_classes, transform=None)

    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=args.num_workers,
                            collate_fn=LoadDataAndLabel.collate_fn)

    val_process_bar = tqdm(val_loader, file=sys.stdout)

    # val
    predictions,true_labels=[],[]
    model.eval()
    with torch.no_grad():
        for _, item in enumerate(val_process_bar):
            data, label = item
            outputs = model.forward(data.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            true_y = torch.max(label, dim=1)[1]
            predictions.extend(predict_y.tolist())
            true_labels.extend(true_y.tolist())
            num_acc += torch.eq(predict_y, true_y.to(device)).sum().item()
            num_sample += outputs.shape[0]
            acc = num_acc / num_sample
            val_process_bar.desc = "accuracy: {:.6f}".format(acc)
        f1 = f1_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)

        result = {"f1 score": f1, "recall": recall, "precision": precision,"accuracy": acc}
    if args.save:
        with open(os.path.join(args.save_path, name, "result.txt"), "w") as f:
            json_str = json.dumps(result, indent=0)
            f.write(json_str)
            f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=r'C:\Users\lengxingcb\Desktop\GraduatioDesignExercise\train.csv',
                        help='data path')
    parser.add_argument("--batch_size", default=128)
    parser.add_argument("--num_classes", default=2)
    parser.add_argument("--model", default='v2')
    parser.add_argument("--weight", default='./save_weights/Undersampling_training0/best.pth')
    parser.add_argument("--save_path", default='./val')
    parser.add_argument("--name", default='test')
    parser.add_argument("--save", default=True)

    parser.add_argument("--num_workers", default=0)

    # random seed
    parser.add_argument("--random-seed", default=0)

    args = parser.parse_args()

    predict(args)
