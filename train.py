import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from network import LaneDataset, LaneDetectionUNet, dice_loss
import logging

logger = logging.getLogger(__name__)

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

def set_seed(seed=0):
    # for reproductibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def loss_bce_dice(logits_bhw, label_bhw, alpha=.5):
    label_bhw = label_bhw.float()
    loss_bce = F.binary_cross_entropy_with_logits(logits_bhw, label_bhw)
    loss_dice = dice_loss(logits_bhw, label_bhw)
    return alpha * loss_bce + (1 - alpha) * loss_dice


def save_checkpoint(model, optimizer, epoch, path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch


if __name__ == "__main__":
    # hyper-parameters
    experiment_name = "v3_bn_dice"
    resume_training = True
    initial_epoch = 11
    SEED = 0
    n_epochs = 15
    lr = 0.001
    batch_size = 32
    save_each = 3
    #-------------------------

    logging.basicConfig(filename=f'checkpoints/{experiment_name}.log', encoding='utf-8', level=logging.DEBUG)

    # Creating datasets
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\image_2"
    gt_folder = r"data\labels"

    data = LaneDataset(img_folder, gt_folder)
    print(f"All training samples: {len(data)}")

    train_data, val_data = random_split(data, [200, 89])
    set_seed(SEED)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)

    model = LaneDetectionUNet()

    # not enough memory in CUDA :(
    # model.to(DEVICE)

    loss_fn = loss_bce_dice

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    if resume_training:
        # load a pre-trained model and resume training
        try:
            load_checkpoint(model, optimizer, f"checkpoints/shallowUNET_{experiment_name}_ep{initial_epoch}.pth")
        except:
            raise FileNotFoundError("Unable to laod trained model to resume training")

    for epoch in range(n_epochs):
        global_epoch = initial_epoch + epoch
        print(f"Training local epoch {epoch + 1}/{n_epochs}")
        model.train()
        epoch_tr_loss = 0.
        for b, batch in enumerate(train_loader):
            img, label = batch

            optimizer.zero_grad()

            logits = model(img)

            # B, 1, H, W
            
            loss = loss_fn(logits.squeeze(1), label.squeeze(1))
            epoch_tr_loss += loss.item()
            loss.backward()

            optimizer.step()
            print(f"   Batch {b} -> loss: {loss.item():.3f}")
        epoch_tr_loss /= (b + 1)

        # validation round
        model.eval()
        epoch_val_loss = 0.
        with torch.no_grad():
            for b, (img, label) in enumerate(val_loader):
                logits = model(img)
                loss = loss_fn(logits, label)
                epoch_val_loss += loss.item()
        epoch_val_loss /= (b + 1)

        if ((global_epoch > 0) & (global_epoch % save_each == 0)):
            save_checkpoint(model, optimizer, epoch,  f"checkpoints/shallowUNET_{experiment_name}_ep{global_epoch}.pth")

        logging.info(f"Global epoch: {global_epoch} -> Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}")
        print(f"Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}")
        print(f"-----------------------------------------------------------------------")
    
    save_checkpoint(model, optimizer, epoch,  f"checkpoints/shallowUNET_{experiment_name}_ep{global_epoch}.pth")