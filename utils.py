import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset import MyDataset
import torch
from torch.utils.data import DataLoader
import numpy as np


MEAN = [178.4296, 176.6811, 174.9059]
STD = [62.4237, 63.3389, 62.5483]


def get_train_transform(width, height):
    return A.Compose([
        A.Resize(width, height),
        A.VerticalFlip(p=0.05),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.Normalize(mean=MEAN, std=STD),
        ToTensorV2()
    ])


def get_val_transform(width, height):
    return A.Compose([
        A.Resize(width, height),
        A.Normalize(mean=MEAN, std=STD),
        A.pytorch.ToTensorV2()
    ])


def get_loaders(img_dir, mask_dir, imsize, batch_size, num_workers=2, pin_memory=True):
    train_transform = get_train_transform(*imsize)
    val_transform = get_val_transform(*imsize)

    train_set = MyDataset(img_dir, mask_dir, train=True, transform=train_transform)
    val_set = MyDataset(img_dir, mask_dir, train=False, transform=val_transform)

    train_loader = DataLoader(
        dataset=train_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        dataset=val_set, 
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        shuffle=False
    )

    return train_loader, val_loader



class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False



def save_checkpoint(model, path, optimizer=None):
    print("==> Saving checkpoint")
    checkpoint = {"state_dict": model.state_dict()}
    if optimizer:
        checkpoint["optimizer"] = optimizer.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(path, model, optimizer):
    print("==> Loading checkpoint")
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])