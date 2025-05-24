import os
import cv2
import tqdm
from network import LaneDataset
from torch.utils.data import random_split
from torchvision.utils import save_image

if __name__ == "__main__":
    train_dir = "training"
    val_dir = "validation"
    n_train, n_val = 200, 89
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\training\image_2"
    gt_folder = r"data\labels"

    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "labels"), exist_ok=True)

    data = LaneDataset(img_folder, gt_folder, augment=False)
    indices = list(range(len(data)))
    train_indices, val_indices = random_split(indices, [n_train, n_val])

    for i in tqdm.tqdm(range(len(data))):
        img_name, gt_name = data.img_gt_list[i]
        dir_ = train_dir if i in train_indices else val_dir

        img = cv2.imread(f"{img_folder}/{img_name}", cv2.IMREAD_UNCHANGED)
        gt = cv2.imread(f"{gt_folder}/{gt_name}", cv2.IMREAD_UNCHANGED)

        cv2.imwrite(os.path.join(dir_, "images", f"{img_name}"), img)
        cv2.imwrite(os.path.join(dir_, "labels", f"{gt_name}"), gt)