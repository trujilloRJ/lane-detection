import cv2
from os import listdir
import numpy as np

# GT_LANE_COLOR = [255, 0, 255]  #BGR
GT_LANE_COLOR = [0, 254, 0]  #BGR

def main():
    gt_folder = r"data\testing" 
    gt_mask_folder = r"data\testing" 
    
    for gt_fn in listdir(gt_folder):
        gt_frame = cv2.imread(f"{gt_folder}/{gt_fn}") #BGR format
        mask = np.all(gt_frame == GT_LANE_COLOR, axis=-1)
        cv2.imwrite(f"{gt_mask_folder}/mask_{gt_fn}", mask.astype(np.uint8))
    

if __name__ == "__main__":
    main()