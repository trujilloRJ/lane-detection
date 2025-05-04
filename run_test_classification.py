import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
from network import LaneDetectionUNet, LaneDataset

if __name__=="__main__":
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\testing\image_2"
    gt_folder = r"data\labels"
    model_name = "shallowUNET_v4_2conv_B_Lmix_R_ep15"

    if not os.path.exists(f"results/{model_name}"):
        os.makedirs(f"results/{model_name}")

    dataset = LaneDataset(img_folder, gt_folder)

    model = LaneDetectionUNet()
    params = torch.load(f"checkpoints/{model_name}.pth")
    model.load_state_dict(params['model_state_dict'])
    model.eval()

    for idx, (img_name, gt_name) in enumerate(dataset.img_gt_list):
        img, _ = dataset[idx]
        img = img[None, :, :, :]
        pred = model(img)
        pred = np.uint8(255*F.sigmoid(pred).squeeze().detach().numpy())
        
        save_path = f"results/{model_name}/{img_name}"
        cv2.imwrite(save_path, pred)

        if idx % 10 == 0:
            print(f"{idx}/{len(dataset)}")