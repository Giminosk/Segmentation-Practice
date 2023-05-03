import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from model import UNetLike
from utils import get_loaders, EarlyStopper, save_checkpoint, load_checkpoint
from metrics import get_metrics
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_DIR = './data/train/'
MASK_DIR = './data/train_masks'
HEIGHT = 160
WIDTH = 240
BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-3

from_checkpoint = False
checkpoint_path = None


def train_step(model, loader, optimizer, loss_func):
    train_total_loss = 0
    for idx, (data, labels) in enumerate(tqdm(loader)):
        data = data.to(DEVICE)
        labels = labels.to(DEVICE).unsqueeze(1).float()
        
        pred = model(data)
        loss = loss_func(pred, labels)
        train_total_loss += loss

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

    return train_total_loss / (idx+1)



def val_step(model, loader, loss_func):
    val_total_loss = 0
    val_total_metrics = torch.Tensor([0, 0, 0, 0])

    model.eval()
    with torch.no_grad():
        for idx, (data, labels) in enumerate(loader):
            data = data.to(DEVICE)
            labels = labels.to(DEVICE).unsqueeze(1).float()
            
            pred = model(data)
            loss = loss_func(pred, labels)
            val_total_loss += loss

            pred = torch.round(torch.sigmoid(pred))
            val_total_metrics += get_metrics(pred.int(), labels.int(), DEVICE)

    model.train()
    val_total_loss /= (idx+1)
    val_total_metrics /= (idx+1)
    
    return {
            'val_loss': val_total_loss,
            'dice': val_total_metrics[0],
            'iou': val_total_metrics[1],
            'cosine': val_total_metrics[2],
            'acc': val_total_metrics[3]
           }



def train():

    train_loader, val_loader = get_loaders(
        img_dir=IMG_DIR, 
        mask_dir=MASK_DIR, 
        imsize=(HEIGHT, WIDTH), 
        batch_size=BATCH_SIZE
    )

    model = UNetLike(in_channels=3, out_channels=1).to(DEVICE)
    optimizer = optim.Adam(params=model.parameters(), lr=LR)
    if from_checkpoint:
        load_checkpoint(checkpoint_path, model, optimizer)

    loss_func = nn.BCEWithLogitsLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=20, verbose=True)
    writer = SummaryWriter('./tensorboard/')
    # early_stopper = EarlyStopper(patience=20, min_delta=10)

    for epoch in range(EPOCHS):

        train_loss = train_step(model, train_loader, optimizer, loss_func)
        print(f'==> Epoch {epoch}: train loss = {train_loss}')
        scheduler.step(train_loss)

        val_stat = val_step(model, val_loader, loss_func)

        print(f'\tValidation: Loss = {round(val_stat["val_loss"].item(), 5)}, Dice coef = {round(val_stat["dice"].item(),5)}, \
IOU = {round(val_stat["iou"].item(), 5)}, Cosine sim = {round(val_stat["cosine"].item(), 5)}, \
Accuracy = {round(val_stat["acc"].item(), 5)}')
              
        writer.add_scalar("Training loss", train_loss, global_step=epoch)
        writer.add_scalar("Validation loss", val_stat["val_loss"].item(), global_step=epoch)
        writer.add_scalar("Dise coefficient", val_stat["dice"].item(), global_step=epoch)
        writer.add_scalar("Cosine similarity", val_stat["cosine"].item(), global_step=epoch)
        writer.add_scalar("Accuracy", val_stat["acc"].item(), global_step=epoch)

        # if early_stopper.early_stop(val_stat["val_loss"].item()):             
        #     break

        if epoch % 20 == 0:
            save_checkpoint(model, f'./checkpoints/Epoch{epoch}.pth.tar', optimizer)


if __name__ == '__main__':
    train()
