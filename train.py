import os
import torch
import torch.nn as nn
import typing as ty
import yaml
import numpy as np
from sklearn.model_selection import KFold
import wandb
from dataset import *
from model import common
from utils import *
# from dataset import *

def model_train(args : ty.Any, config: ty.Dict[str, ty.List[str]]) -> None:
    
    """
    args have train params, config have model params
    check yaml, main file
    """

    # scp /Users/qkrdnjsrl/downloads/chart.png psboys@202.30.30.19:/home/psboys/private/Research/report/

    seed_everything(0)
    X_data, target = load_dataset(args)
    kf = KFold(n_splits = 5, shuffle = True, random_state = 0)
    count = 0
    for train_index, test_index in kf.split(X_data):
        count += 1
        X_train, X_test = X_data.loc[train_index].reset_index(), X_data.loc[test_index].reset_index()
        Y_train, Y_test = target.loc[train_index].reset_index(), target.loc[test_index].reset_index()

        train_dataloader, test_dataloader = get_dataloader(X_train, Y_train), get_dataloader(X_test, Y_test)
        model = common.load_model(args, config)
        print("loaded model..")
        print("run..")
        run(model, train_dataloader, test_dataloader, args, config, count)
            
def run(model, train_dataloader, test_dataloader, args, config, count):
    
    wandb.init( name = config["model"] + "_700_" + str(count), 
                project = "energy efficiency" + "_" + str(args.target), reinit = True)

    model.to(args.device)
    model = nn.DataParallel(model)

    optimizer = get_optimizer(model, config)
    loss_fn = get_loss(args)
    
    if args.task == "regression":  # RMSE
        best_valid = 1e10
    else:       # Accuracy
        best_valid = 0
    
    for epoch in range(int(config["epochs"])):
        train_loss_score = 0

        train_pred, valid_pred = np.array([]), np.array([])
        train_label, valid_label = np.array([]), np.array([])

        model.train()       # Train DataLoader
        for X_data, y_label in train_dataloader:        # Train
            optimizer.zero_grad()
            X_data, y_label = X_data.to(args.device), y_label.to(args.device)
            y_pred = model(x_num = X_data, x_cat = None)
            if args.task == "regression":
                loss = loss_fn(y_pred.to(torch.float64).squeeze(1), y_label.to(torch.float64).squeeze(1))
            else:
                pass

            loss.backward()
            optimizer.step()
            train_loss_score += loss.item()

            if args.task == "regression":
                train_pred = np.append(train_pred, y_pred.cpu().detach().numpy())
                train_label = np.append(train_label, y_label.cpu().detach().numpy())
        
        model.eval()        # Valid DataLoader
        for X_data, y_label in test_dataloader:
            
            X_data, y_label = X_data.to(args.device), y_label.to(args.device) 
            y_pred = model(x_num = X_data, x_cat = None)

            if args.task == "regression":
                valid_pred = np.append(valid_pred, y_pred.cpu().detach().numpy())
                valid_label = np.append(valid_label, y_label.cpu().detach().numpy())

        if args.task == "regression":
            train_score, valid_score = get_rmse_score(train_pred, train_label), get_rmse_score(valid_pred, valid_label)

            if best_valid > valid_score:
                best_valid = valid_score
                config["valid_rmse"] = valid_score

                save_mode_with_json(model, config, args, count)

        wandb.log({
            "train_score" : train_score,
            "train_loss" : train_loss_score,
            "valid_score" : valid_score,
        })
    





        







    
    
    