import cv2
import numpy as np

GT_LANE_COLOR = [255, 0, 255]  #BGR

def main():
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\image_2"
    gt_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\gt_image_2" 
    img_index = "000000"

    frame = cv2.imread(f"{img_folder}/um_{img_index}.png")
    gt_frame = cv2.imread(f"{gt_folder}/um_lane_{img_index}.png") #BGR format
    
    mask = np.all(gt_frame == GT_LANE_COLOR, axis=-1)
    gt_frame[~mask] = [0, 0, 0]

    frame = cv2.addWeighted(frame, 1, gt_frame, 0.3, 0)

    cv2.imshow("Image + lane", frame)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()