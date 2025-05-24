import cv2
import torch
import torch.nn.functional as F
import albumentations as A

from torchvision.io import read_image
from os import listdir
from torch import nn
from torch.utils.data import Dataset, DataLoader
from constants import IMG_HEIGHT, IMG_WIDTH

class LaneDataset(Dataset):
    def __init__(self, img_folder: str, gt_folder: str, augment=False, seed=0):
        self.img_folder = img_folder
        self.gt_folder = gt_folder

        img_name_lists = listdir(img_folder)
        gt_name_lists = listdir(gt_folder)
        self.img_gt_list = [(img, gt) for img, gt in zip(img_name_lists, gt_name_lists)]

        if augment:
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    # A.Rotate(limit=5., p=1.0),
                    # A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=1.0),
                    A.ToTensorV2(),
                ],
                seed=seed
            )
        else:
            self.transform = A.ToTensorV2()

    def __len__(self):
        return len(self.img_gt_list)
    
    def __getitem__(self, idx):
        img_fn, gt_fn = self.img_gt_list[idx]
        img = cv2.imread(f"{self.img_folder}/{img_fn}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        gt = cv2.imread(f"{self.gt_folder}/{gt_fn}", cv2.IMREAD_GRAYSCALE)

        # Pad or crop img and gt to IMG_HEIGHT, IMG_WIDTH
        h, w = img.shape[:2]
        pad_h = max(0, IMG_HEIGHT - h)
        pad_w = max(0, IMG_WIDTH - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        gt = cv2.copyMakeBorder(gt, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        img = img[:IMG_HEIGHT, :IMG_WIDTH]
        gt = gt[:IMG_HEIGHT, :IMG_WIDTH]

        # if augmented is False, this will be just cast as tensors
        augmented = self.transform(image = img, mask = gt)
        img = augmented['image']
        gt = augmented['mask']

        return img, gt
    
    def get_curr_img_fn(self, idx):
        img_fn, _ = self.img_gt_list[idx]
        return img_fn
    
    def subset_on_indices(self, indices: list):
        self.img_gt_list = [self.img_gt_list[i] for i in indices]


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, double_conv: bool = False):
        super().__init__()
        # padding = 1 preserve the image size after the convolution

        if double_conv:
            self.operation = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        else:
            self.operation = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )

    def forward(self, X):
        return self.operation(X)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor=2, out_size=None, double_conv: bool = False):
        super().__init__()

        if out_size:
            uplayer = nn.Upsample(out_size, mode="bilinear", align_corners=True)
        else:
            uplayer = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        if double_conv:
            self.operation = nn.Sequential(
                uplayer,
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU()
            )
        else:
            self.operation = nn.Sequential(
                uplayer,
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
            )

    def forward(self, X):
        return self.operation(X)


class LaneDetectionUNet(nn.Module):
    def __init__(self, double_conv=False):
        super().__init__()

        # chs = [16, 32, 64]
        chs = [32, 64, 128]

        self.d1 = Down(3, chs[0], double_conv)  # f=1/2
        self.d2 = Down(chs[0], chs[1], double_conv) # f=1/4
        self.d3 = Down(chs[1], chs[2], double_conv) # f=1/8
        self.u1 = Up(chs[2], chs[1], double_conv=double_conv)   # f=1/4
        self.u2 = Up(chs[2], chs[0], double_conv=double_conv)   # f=1/2
        self.u3 = Up(chs[1], chs[0], out_size=(IMG_HEIGHT, IMG_WIDTH), double_conv=double_conv)   # f=1
        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

    def forward(self, X):

        X1 = self.d1(X)
        X2 = self.d2(X1)
        X = self.d3(X2)

        X = self.u1(X)

        # concat X2 and X along channels
        X = self.concatenate_tensors(X, X2)
        X = self.u2(X)

        # concat X1 and X along channels
        X = self.concatenate_tensors(X, X1)
        X = self.u3(X)

        logits = self.out_conv(X)
        return logits
    
    def concatenate_tensors(self, X, X1):
        diffY = X1.size()[2] - X.size()[2]
        diffX = X1.size()[3] - X.size()[3]
        X = F.pad(X, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        X = torch.cat([X, X1], dim=1)
        return X
    

def dice_loss(pred_bhw, target_bhw, eps=0.001, **kwargs):
    pred_bhw = torch.sigmoid(pred_bhw) 
    sum_dim = (-1, -2) # sum over H, W
    intersection = (pred_bhw * target_bhw).sum(dim=sum_dim) 
    dice = (2.0 * intersection + eps) / (pred_bhw.sum(dim=sum_dim) + target_bhw.sum(dim=sum_dim) + eps)
    return 1.0 - dice.mean()

def jaccard_loss(pred_bhw, target_bhw, eps=0.001):
    pred_bhw = torch.sigmoid(pred_bhw) 
    sum_dim = (-1, -2) # sum over H, W
    intersection = (pred_bhw * target_bhw).sum(dim=sum_dim) 
    dice = (intersection + eps) / (pred_bhw.sum(dim=sum_dim) + target_bhw.sum(dim=sum_dim) + eps - intersection)
    return 1.0 - dice.mean()

def loss_bce_dice(logits_bhw, label_bhw, wbce, alpha=.5):
    label_bhw = label_bhw.float()
    loss_bce = F.binary_cross_entropy_with_logits(logits_bhw, label_bhw, weight=wbce)
    loss_dice = dice_loss(logits_bhw, label_bhw)
    return loss_bce + loss_dice, loss_dice

