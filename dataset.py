import torch
import pandas as pd
import typing as ty
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold, train_test_split

def load_dataset(args : ty.Any) -> None:
    X_data = pd.read_excel(args.path)
    
    if args.target == 1:
        target = X_data["Y1"]
        X_data.drop(["Y1", "Y2"], axis = 1, inplace = True)
    elif args.target == 2:
        target = X_data["Y2"]
        X_data.drop(["Y1", "Y2"], axis = 1, inplace = True)
    else:
        pass

    return X_data, target

class customDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target
    
    def __getitem__(self, idx):
        x_data = torch.FloatTensor(self.data.loc[idx].to_numpy())
        y_data = torch.FloatTensor(self.target.loc[idx].to_numpy())

        return x_data, y_data
    
    def __len__(self):
        return len(self.data)

    
def get_dataloader(x_data, y_data):
    dataset = customDataset(x_data, y_data)
    data_dataloader = DataLoader(dataset, batch_size = 32, pin_memory = True)
    return data_dataloader
