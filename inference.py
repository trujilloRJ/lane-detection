import numpy as np
import torch
import cv2
import torch.nn.functional as F
from network import LaneDetectionUNet, LaneDataset
from torch.utils.data import DataLoader

KEY_ESC = 27
KEY_SPACE = 32
KEY_A = 97
KEY_D = 100
KEY_N = 110
KEY_M = 109
KEY_G = 103
KEY_S = 115

if __name__=="__main__":
    # img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\image_2"
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\testing\image_2"
    gt_folder = r"data\labels"

    dataset = LaneDataset(img_folder, gt_folder)

    model = LaneDetectionUNet()
    params = torch.load("checkpoints/shallowUNET_v3_bn_dice_ep25.pth")
    model.load_state_dict(params['model_state_dict'])
     
    model.eval()

    run = True
    img_index = 0
    thr = 0.5
    while(run):
        img, gt = dataset[img_index]
        
        img = img[None, :, :, :]
        pred = model(img)
        pred = np.uint8(255*F.sigmoid(pred).squeeze().detach().numpy())
        pred[pred > 255*thr] = 255
        pred[pred < 255*thr] = 0
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

        gt = np.uint8(255*gt.squeeze().detach().numpy())
        gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

        img = torch.swapaxes(img.squeeze(), 0, 2)
        img = torch.swapaxes(img.squeeze(), 0, 1)
        img = np.uint8(img.detach().numpy())

        # making pred blue and gt green
        pred[:, :, 1:] = 0
        gt[:, :, 0] = 0 
        gt[:, :, 2] = 0 

        frame1 = cv2.addWeighted(img, 1, gt, 0.5, 0)
        frame2 = cv2.addWeighted(img, 1, pred, 0.5, 0)

        # cv2.imshow(f"GT", frame1)
        cv2.imshow(f"Prediction", frame2)
        key = cv2.waitKey(0)

        if key == KEY_D:
            img_index += 1
        if key == KEY_A:
            img_index -= 1
        if key == KEY_ESC:
            run = False
        else:
            print(key)


    cv2.destroyAllWindows()
