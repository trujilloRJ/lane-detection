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
                    # A.HorizontalFlip(p=0.5), # v1
                    A.RGBShift(r_shift_limit=20, g_shift_limit=20, b_shift_limit=20, p=0.5), # v2
                    # A.ToGray(p=0.5), # v3
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
        if h != IMG_HEIGHT or w != IMG_WIDTH:
            pass
        pad_h = max(0, IMG_HEIGHT - h)
        pad_w = max(0, IMG_WIDTH - w)
        img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        img = img[:IMG_HEIGHT, :IMG_WIDTH]

        # if augmented is False, this will be just cast as tensors
        augmented = self.transform(image = img, mask = gt)
        img = augmented['image']
        gt = augmented['mask']

        return img.float(), gt.float()
    
    def get_curr_img_fn(self, idx):
        img_fn, _ = self.img_gt_list[idx]
        return img_fn
    
    def subset_on_indices(self, indices: list):
        self.img_gt_list = [self.img_gt_list[i] for i in indices]


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size = 3, padding = "same", use_batch_norm = True, **kwargs):
        super().__init__()
        use_bias = not use_batch_norm

        self.operation = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=use_bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, X):
        return self.operation(X)


class UpBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, scale_factor=2, out_size=None):
        super().__init__()

        if out_size:
            self.uplayer = nn.Upsample(out_size, mode="bilinear", align_corners=True)
        else:
            self.uplayer = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=True)

        self.convlayer = nn.Sequential(
            ConvBlock(in_ch, in_ch//2),
            ConvBlock(in_ch//2, out_ch),
        )

    def forward(self, X, X1):
        X = self.uplayer(X)
        X = self.concatenate_tensors(X, X1)
        return self.convlayer(X)

    @staticmethod
    def concatenate_tensors(X, X1):
        diffY = X1.size()[2] - X.size()[2]
        diffX = X1.size()[3] - X.size()[3]
        X = F.pad(X, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        X = torch.cat([X, X1], dim=1)
        return X
        

class LaneDetectionUNet(nn.Module):
    def __init__(self, wide=False):
        super().__init__()

        if wide:
            chs = [32, 64, 128]
        else:
            chs = [16, 32, 64]

        self.enc = nn.Sequential(
            ConvBlock(3, chs[0]),
            ConvBlock(chs[0], chs[0]),
        )

        self.d1 = self._create_down_block(chs[0], chs[1])
        self.d2 = self._create_down_block(chs[1], chs[2]//2)
        self.u1 = UpBlock(chs[2], chs[1]//2)
        self.u2 = UpBlock(chs[1], chs[0])

        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

    def _create_down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, X):
        X1 = self.enc(X)
        X2 = self.d1(X1)
        X = self.d2(X2)

        X = self.u1(X, X2)
        X = self.u2(X, X1)

        logits = self.out_conv(X)
        return logits
    

class LaneDetectionDeeperUNet(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        chs = [32, 64, 128, 256]

        self.enc = nn.Sequential(
            ConvBlock(3, chs[0]),
            ConvBlock(chs[0], chs[0]),
        )

        self.d1 = self._create_down_block(chs[0], chs[1])
        self.d2 = self._create_down_block(chs[1], chs[2])
        self.d3 = self._create_down_block(chs[2], chs[3]//2)
        self.u1 = UpBlock(chs[3], chs[2]//2)
        self.u2 = UpBlock(chs[2], chs[1]//2)
        self.u3 = UpBlock(chs[1], chs[0])

        self.out_conv = nn.Conv2d(chs[0], 1, kernel_size=1)

    def _create_down_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_ch, out_ch),
            ConvBlock(out_ch, out_ch),
        )

    def forward(self, X):
        X1 = self.enc(X)
        X2 = self.d1(X1)
        X3 = self.d2(X2)
        X = self.d3(X3)

        X = self.u1(X, X3)
        X = self.u2(X, X2)
        X = self.u3(X, X1)

        logits = self.out_conv(X)
        return logits
    

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


if __name__ == "__main__":
    X = torch.rand(1, 3, 400, 400)
    # model = LaneDetectionUNet(wide=True)
    model = LaneDetectionDeeperUNet(wide=True)
    print(model)
    logits = model(X)
    print(logits.shape)