import numpy as np
import torch
import cv2
import torch.nn.functional as F
from network import LaneDetectionUNet, LaneDataset
from torch.utils.data import DataLoader

if __name__=="__main__":
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\image_2"
    gt_folder = r"data\labels"

    dataset = LaneDataset(img_folder, gt_folder)

    model = LaneDetectionUNet()
    model.load_state_dict(torch.load("checkpoints/shallowUNET_ep9.pth"))
    model.eval()

    img, gt = dataset[0]

    thr = 0.5
    img = img[None, :, :, :]
    pred = model(img)
    pred = np.uint8(255*F.sigmoid(pred).squeeze().detach().numpy())
    pred[pred > 255*thr] = 255
    pred[pred < 255*thr] = 0

    cv2.imshow("Pred", pred)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
