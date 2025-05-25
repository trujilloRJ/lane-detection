import cv2
import os
import tqdm
import numpy as np
from constants import IMG_HEIGHT, IMG_WIDTH

GT_LANE_COLOR = [255, 0, 255]  #BGR
# GT_LANE_COLOR = [0, 254, 0]  #BGR

def main():
    gt_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\split_dataset\validation\labels" 
    gt_mask_folder = r"data\labels\validation"

    os.makedirs(gt_mask_folder, exist_ok=True)
    
    for gt_fn in tqdm.tqdm(os.listdir(gt_folder)):
        gt_frame = cv2.imread(f"{gt_folder}/{gt_fn}") #BGR format
        mask = np.all(gt_frame == GT_LANE_COLOR, axis=-1)

        if gt_frame.shape[0] != IMG_HEIGHT or gt_frame.shape[1] != IMG_WIDTH:
            padded = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=mask.dtype)
            h = min(IMG_HEIGHT, mask.shape[0])
            w = min(IMG_WIDTH, mask.shape[1])
            padded[:h, :w] = mask[:h, :w]
            mask = padded

        cv2.imwrite(f"{gt_mask_folder}/{gt_fn}", mask.astype(np.uint8))
    

if __name__ == "__main__":
    main()