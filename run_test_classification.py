import cv2
import os
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from network import LaneDataset, make_unet_from_file
from dotenv import dotenv_values


if __name__=="__main__":
    env = dotenv_values(".env")
    img_folder = os.path.join(env["VALIDATION_DATA_PATH"], "images")
    dummy = r"data\labels\validation"
    epoch = "64"
    model_name = "BUnet_d3_c32_a0_SOneCycle"
    exp_name = f"{model_name}_ep{epoch}"

    os.makedirs(f"results/{model_name}", exist_ok=True)

    dataset = LaneDataset(img_folder, dummy)

    model, config = make_unet_from_file(f"checkpoints/{model_name}_config.json")
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