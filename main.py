import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3, 4, 5" 
import yaml
import typing as ty
import argparse
import torch
from train import model_train

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type = str, required = True)        # train
    parser.add_argument("--data", type = str, required = True)          # datapath
    parser.add_argument("--task", type = str, required = True)          # regression
    parser.add_argument("--target", type = int, required = True)
    parser.add_argument("--model", type = str, required = True)
    parser.add_argument("--fold", type = int, required = True)
    parser.add_argument("--savepath", type = str, required = True)
    # python main.py --action train --data ENB2012_data.xlsx --task regression --target 1 --model fttransformer --fold 5 --savepath output
    args = parser.parse_args()
    args.path = os.path.join("data", args.data)
    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    with open(f"yaml/{args.model}.yaml") as f:
        config = yaml.load(f, Loader = yaml.FullLoader)[args.model]

    if args.action == "train":
        model_train(args, config)
