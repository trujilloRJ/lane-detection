import cv2
import torch
import torch.nn.functional as F

from torchvision.io import read_image
from os import listdir
from torch import nn
from torch.utils.data import Dataset, DataLoader

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
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
        return img.float(), gt.float()


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # padding = 1 preserve the image size after the convolution
        self.operation = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
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

        self.out_conv = nn.Conv2d(16, 1, kernel_size=3, padding=1)

        self.out = nn.Softmax()

    def forward(self, X):
        X = X[None, :, :, :] # add minibatch dimension, REMOVE later

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

        X = self.out(self.out_conv(X))
        return X


if __name__ == "__main__":
    # print(f"Using {DEVICE} device")
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\image_2"
    gt_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\gt_image_2"

    train_data = LaneDataset(img_folder, gt_folder)

    img, gt = train_data[0]

    net = LaneDetectionUNet()
    img_out = net(img)
    
    img_out = img_out.squeeze().detach().numpy()
    cv2.imshow("Image + lane", img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

