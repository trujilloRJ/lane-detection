import cv2
import torch
import torch.nn.functional as F

from torchvision.io import read_image
from os import listdir
from torch import nn
from torch.utils.data import Dataset, DataLoader

# not all images have this size, but they differ from 1 or 2 pixels, we will need to crop or pad accordingly 
IMG_HEIGHT = 375  
IMG_WIDTH = 1242

class LaneDataset(Dataset):
    def __init__(self, img_folder: str, gt_folder: str):
        self.img_folder = img_folder
        self.gt_folder = gt_folder

        img_name_lists = listdir(img_folder)
        gt_name_lists = listdir(gt_folder)
        self.img_gt_list = [(img, gt) for img, gt in zip(img_name_lists, gt_name_lists)]

    def __len__(self):
        return len(self.img_gt_list)
    
    def __getitem__(self, idx):
        img_fn, gt_fn = self.img_gt_list[idx]
        img = read_image(f"{self.img_folder}/{img_fn}")
        gt = read_image(f"{self.gt_folder}/{gt_fn}")

        # TODO: this can be done offline for speeding-up training
        # Crop or pad the image and ground truth to the target size
        img = F.pad(img, (0, max(0, IMG_WIDTH - img.shape[2]), 0, max(0, IMG_HEIGHT - img.shape[1])), mode='constant', value=0)
        gt = F.pad(gt, (0, max(0, IMG_WIDTH - gt.shape[2]), 0, max(0, IMG_HEIGHT - gt.shape[1])), mode='constant', value=0)
        img = img[:, :IMG_HEIGHT, :IMG_WIDTH]
        gt = gt[:, :IMG_HEIGHT, :IMG_WIDTH]

        return img.float(), gt.float()
    
    def get_curr_img_fn(self, idx):
        img_fn, _ = self.img_gt_list[idx]
        return img_fn


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # padding = 1 preserve the image size after the convolution
        self.operation = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_ch),
            nn.MaxPool2d(2)
        )
    def forward(self, X):
        return self.operation(X)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor=2, out_size=None):
        super().__init__()

        if out_size:
            uplayer = nn.Upsample(out_size, mode="bilinear", align_corners=True)
        else:
            uplayer = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        self.operation = nn.Sequential(
            uplayer,
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
    def forward(self, X):
        return self.operation(X)


class LaneDetectionUNet(nn.Module):
    def __init__(self):
        super().__init__()

        # TODO: add batch normalization
        self.d1 = Down(3, 16)  # f=1/2
        self.d2 = Down(16, 32) # f=1/4
        self.u1 = Up(32, 16)   # f=1/2
        self.u2 = Up(32, 16, out_size=(IMG_HEIGHT, IMG_WIDTH))   # f=1

        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, X):

        X1 = self.d1(X)
        X = self.d2(X1)
        X = self.u1(X)

        # concat X1 and X along channels
        diffY = X1.size()[2] - X.size()[2]
        diffX = X1.size()[3] - X.size()[3]
        X = F.pad(X, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        X = torch.cat([X, X1], dim=1)
        
        X = self.u2(X)

        logits = self.out_conv(X)
        return logits


def dice_loss(pred_bhw, target_bhw, eps=0.001):
    pred_bhw = torch.sigmoid(pred_bhw) 

    sum_dim = (-1, -2) # sum over H, W

    intersection = (pred_bhw * target_bhw).sum(dim=sum_dim) 
    dice = (2.0 * intersection + eps) / (pred_bhw.sum(dim=sum_dim) + target_bhw.sum(dim=sum_dim) + eps)

    return 1.0 - dice.mean()

