import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from network import LaneDataset, LaneDetectionUNet, dice_loss

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


if __name__ == "__main__":
    # hyper-parameters
    experiment_name = "v2_dice_loss"
    SEED = 0
    n_epochs = 12
    lr = 0.001
    batch_size = 32
    save_each = 3
    #-------------------------

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

    for epoch in range(n_epochs):
        print(f"Training epoch {epoch + 1}/{n_epochs}")
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

        if ((epoch > 0) & (epoch % save_each == 0)) | (epoch == (n_epochs - 1)):
            torch.save(model.state_dict(), f"checkpoints/shallowUNET_{experiment_name}_ep{epoch}.pth")

        print(f"Train loss: {epoch_tr_loss:.3f} | Validation loss: {epoch_val_loss:.3f}")
        print(f"-----------------------------------------------------------------------")