import torch
import logging
import tqdm
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from network import LaneDataset, LaneDetectionUNet, loss_bce_dice, dice_loss
from enum import Enum
import json

logger = logging.getLogger(__name__)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def set_seed(seed=0):
    # for reproductibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)

class OptimizerChoice(Enum):
    ADAMW = "adamw"
    SGD = "sgd"

def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


if __name__ == "__main__":
    # hyper-parameters
    experiment_name = "UNet2down_v9_Srop_adam_augv2"
    resume_training = False
    initial_epoch = 0
    SEED = 0
    n_epochs = 150
    lr = 3e-4
    batch_size = 2
    save_each = 25
    optimizer_choice = OptimizerChoice.ADAMW
    loss_fn = loss_bce_dice
    wbce = torch.tensor([0.8], device=DEVICE) # weight of the BCE loss
    wide = True
    augment_data = True
    #-------------------------

    set_seed(SEED)

    # initializing experiment configuration
    config = {
        "exp_name": experiment_name,
        "optimizer_choice": optimizer_choice.value,
        "augmentation": augment_data,
        "wide": wide
    }

    logging.basicConfig(filename=f'checkpoints/{experiment_name}.log', encoding='utf-8', level=logging.DEBUG)

    # Creating datasets
    train_img_dir = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\split_dataset\training\images"
    val_img_dir = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\split_dataset\validation\images"
    train_gt_dir = r"data\labels\training"
    val_gt_dir = r"data\labels\validation"

    train_data = LaneDataset(train_img_dir, train_gt_dir, augment=augment_data)
    val_data = LaneDataset(val_img_dir, val_gt_dir, augment=False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = LaneDetectionUNet(double_conv = True, wide = wide)

    model.to(DEVICE)
  
    if optimizer_choice is OptimizerChoice.ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_choice is OptimizerChoice.SGD:
        # momentum value from UNET paper
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_choice}")
    
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*n_epochs, eta_min=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience=5)

    # load a pre-trained model and resume training
    if resume_training:
        try:
            load_checkpoint(model, optimizer, f"checkpoints/{experiment_name}_ep{initial_epoch}.pth")
        except:
            raise FileNotFoundError("Unable to load trained model to resume training")

    lower_val_loss = 100
    for epoch in range(n_epochs):
        global_epoch = initial_epoch + epoch + 1
        print(f"Training local epoch {epoch + 1}/{n_epochs}")
        model.train()
        epoch_tr_loss = 0.
        for b, batch in enumerate(tqdm.tqdm(train_loader, desc=f"Training epoch {epoch}")):
            img, label = batch
            img, label = img.to(DEVICE), label.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(img)
            
            logits, label = logits.squeeze(1), label.squeeze(1).float()

            loss, _ = loss_fn(logits, label, wbce=wbce)

            epoch_tr_loss += loss.item()
            loss.backward()

            optimizer.step()

        epoch_tr_loss /= (b + 1)

        # validation round
        model.eval()
        epoch_val_loss = 0.
        epoch_val_dice_loss = 0.
        with torch.no_grad():
            for b, (img, label) in enumerate(val_loader):
                img = img.to(DEVICE)
                label = label.to(DEVICE)
                logits = model(img)
                logits, label = logits.squeeze(1), label.squeeze(1).float()
                loss, loss_dice = loss_fn(logits, label, wbce=wbce)
                epoch_val_loss += loss.item()
                epoch_val_dice_loss += loss_dice.item()
        epoch_val_loss /= (b + 1)
        epoch_val_dice_loss /= (b + 1)

        # scheduler step (epoch-wise, for ROP)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if scheduler.get_last_lr()[0] > 5e-6:
                scheduler.step(epoch_val_dice_loss)

        if epoch_val_loss < lower_val_loss:
            lower_val_loss = epoch_val_loss
            save_this = True
        else:
            save_this = False

        if ((global_epoch % save_each == 0) | (save_this)):
            save_checkpoint(model, optimizer, epoch,  f"checkpoints/{experiment_name}_ep{global_epoch}.pth")

        logging.info(f"Global epoch: {global_epoch} -> Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}")
        print(f"Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"-----------------------------------------------------------------------")
    
    with open(f'checkpoints/{experiment_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)