import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from network import LaneDetectionUNet, LaneDataset

if __name__=="__main__":
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\split_dataset\validation\images"
    dummy = r"data\labels\validation"
    epoch = "100"
    model_name = "sUNet_v7_Srop_adam_augv0"
    exp_name = f"{model_name}_ep{epoch}"

    os.makedirs(f"results/{model_name}", exist_ok=True)

    dataset = LaneDataset(img_folder, dummy)

    model = LaneDetectionUNet(double_conv=True)
    params = torch.load(f"checkpoints/{exp_name}.pth")
    model.load_state_dict(params['model_state_dict'])
    model.eval()

    for idx, (img_name, _) in tqdm.tqdm(enumerate(dataset.img_gt_list)):
        img, _ = dataset[idx]
        img = img[None, :, :, :]
        pred = model(img)
        pred = np.uint8(255*F.sigmoid(pred).squeeze().detach().numpy())
        
        save_path = f"results/{model_name}/{img_name}"
        cv2.imwrite(save_path, pred)