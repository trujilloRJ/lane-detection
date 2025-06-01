import torch
import logging
import tqdm
import json
import numpy as np
from torch.utils.data import DataLoader
from network import LaneDataset, LaneDetectionUNet, LaneDetectionDeeperUNet, loss_bce_dice
from enum import Enum
from common import set_seed
from runner import train_step, validate_epoch, load_checkpoint, save_checkpoint

logger = logging.getLogger(__name__)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class OptimizerChoice(Enum):
    ADAMW = "adamw"
    SGD = "sgd"

def train_model(initial_epoch, n_epochs, model, train_loader, val_loader, optimizer, scheduler, loss_fn, wbce):
    lower_val_loss = 100
    for epoch in range(n_epochs):
        global_epoch = initial_epoch + epoch + 1
        print(f"Training local epoch {epoch + 1}/{n_epochs}")

        model.train()
        epoch_tr_loss = 0.
        for batch in tqdm.tqdm(train_loader, desc=f"Training epoch {epoch + 1}"):
            img, label = batch
            img, label = img.to(DEVICE), label.to(DEVICE)

            tr_loss = train_step(model, optimizer, loss_fn, img, label, wbce)
            epoch_tr_loss += tr_loss

            if (isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR) 
                | (isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR))):
                scheduler.step()

        epoch_tr_loss /= len(train_loader)

        epoch_val_loss = validate_epoch(model, val_loader, loss_fn, wbce, DEVICE)

        # scheduler step epoch-wise
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if scheduler.get_last_lr()[0] > 5e-6:
                scheduler.step(epoch_val_loss)

        if epoch_val_loss < lower_val_loss:
            lower_val_loss = epoch_val_loss
            save_this = (epoch/n_epochs > 0.1)
        else:
            save_this = False

        if ((global_epoch % save_each == 0) | (save_this)):
            save_checkpoint(model, optimizer, epoch,  f"checkpoints/{experiment_name}_ep{global_epoch}.pth")

        logging.info(f"Global epoch: {global_epoch} -> Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}")
        print(f"Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        print(f"-----------------------------------------------------------------------")

def exp_lr(m, start_lr, end_lr, N):
    lr_vec = start_lr * (end_lr/start_lr)**(m/(N-1))
    return lr_vec

def lr_range_test(n_iter, start_lr, end_lr, optimizer, n_acc_steps, model, train_loader, loss_fn, wbce):
    loss_lr = np.zeros((n_iter, 2))
    steps = 0
    iter_ = 0
    epoch_tr_loss = 0.
    while iter_ < n_iter:
        model.train()
        for batch in tqdm.tqdm(train_loader, desc=f"Training..."):
            img, label = batch
            img, label = img.to(DEVICE), label.to(DEVICE)

            tr_loss = train_step(model, optimizer, loss_fn, img, label, wbce)
            epoch_tr_loss += tr_loss

            steps += 1

            if steps >= n_acc_steps:
                # save loss and lr
                epoch_tr_loss /= n_acc_steps
                lr = optimizer.param_groups[0]['lr']
                loss_lr[iter_, 0] = lr
                loss_lr[iter_, 1] = epoch_tr_loss
                print(f"LR: {lr} | Train loss: {epoch_tr_loss:.3f}")

                # udpate and step lr
                epoch_tr_loss = 0
                iter_ += 1
                steps = 0
                new_lr = start_lr * (end_lr/start_lr)**(iter_/(n_iter-1))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr

            if iter_ >= n_iter: break

    return loss_lr

if __name__ == "__main__":
    # hyper-parameters
    experiment_name = "UNet3down_v10_Scos_adam_augv2"
    resume_training = False
    initial_epoch = 0
    SEED = 0
    n_epochs = 70
    lr = 5e-4
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

    model = LaneDetectionDeeperUNet(wide = wide)

    model.to(DEVICE)
  
    if optimizer_choice is OptimizerChoice.ADAMW:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer_choice is OptimizerChoice.SGD:
        # momentum value from UNET paper
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, weight_decay=1e-4)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_choice}")
    
    # values drawn from lr range test
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader)*n_epochs, eta_min=5e-6)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience=5)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=2e-3, total_steps=n_epochs*len(train_loader))

    # load a pre-trained model and resume training
    if resume_training:
        load_checkpoint(model, optimizer, f"checkpoints/{experiment_name}_ep{initial_epoch}.pth")

    train_model(initial_epoch, n_epochs, model, train_loader, val_loader, optimizer, scheduler, loss_fn, wbce)

    with open(f'checkpoints/{experiment_name}_config.json', 'w') as f:
        json.dump(config, f, indent=4)

    # n_iter = 100
    # start_lr = 1e-6
    # end_lr = 1
    # n_acc_steps = 16  # backwards pass before update lr
    # for param_group in optimizer.param_groups:
    #     param_group['lr'] = start_lr
    # loss_lr = lr_range_test(n_iter, start_lr, end_lr, optimizer, n_acc_steps, model, train_loader, loss_fn, wbce)
    # np.save("lr_range_test.npy", loss_lr)
    