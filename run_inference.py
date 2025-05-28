import numpy as np
import torch
import cv2
import torch.nn.functional as F
import json
from network import LaneDetectionUNet, LaneDataset, loss_bce_dice, dice_loss, jaccard_loss
from common import compute_tp_fp_fn, pad_gt
from constants import IMG_HEIGHT, IMG_WIDTH


KEY_ESC = 27
KEY_SPACE = 32
KEY_A = 97
KEY_D = 100
KEY_N = 110
KEY_M = 109
KEY_G = 103
KEY_S = 115

if __name__=="__main__":
    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\split_dataset\validation\images"
    gt_folder = r"data\labels\validation"
    wbce = torch.tensor([0.8]) # weight of the BCE loss
    model_name = "sUNetW_v8_Srop_adam_augv2"
    epoch = "68"
    exp_name = f"{model_name}_ep{epoch}"
    wide = True

    dataset = LaneDataset(img_folder, gt_folder)

    model = LaneDetectionUNet(double_conv=True, wide=wide)
    params = torch.load(f"checkpoints/{exp_name}.pth")
    model.load_state_dict(params['model_state_dict'])
     
    model.eval()

    run = True
    img_index = 0
    thr = 0.2
    while(run):
        img, gt = dataset[img_index]
        gt = gt.squeeze()
        
        img = img[None, :, :, :]
        pred = model(img)
        logits_bhw = pred.squeeze()
        loss_mixed, loss_dice = loss_bce_dice(logits_bhw, gt, wbce)
        loss_bce = F.binary_cross_entropy_with_logits(logits_bhw, gt, weight=wbce)
        loss_jaccard = jaccard_loss(logits_bhw, gt)

        # computing metrics
        gt = gt.detach().numpy().astype(float)
        gt = pad_gt(gt, IMG_HEIGHT, IMG_WIDTH)
        probs = F.sigmoid(pred).squeeze().detach().numpy()
        probs[probs > thr] = 1.
        probs[probs < thr] = 0.
        tp, fp, fn = compute_tp_fp_fn(probs, gt)
        precision = tp/(tp + fp)
        recall = tp/(tp + fn)

        # visualization
        pred = np.uint8(255*F.sigmoid(pred).squeeze().detach().numpy())
        pred[pred > 255*thr] = 255
        pred[pred < 255*thr] = 0
        pred = cv2.cvtColor(pred, cv2.COLOR_GRAY2BGR)

        gt = np.uint8(255*gt)
        gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)

        img = torch.swapaxes(img.squeeze(), 0, 2)
        img = torch.swapaxes(img.squeeze(), 0, 1)
        img = np.uint8(img.detach().numpy())
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # making pred blue and gt green
        pred[:, :, 1:] = 0
        gt[:, :, 0] = 0 
        gt[:, :, 2] = 0 

        frame1 = cv2.addWeighted(img, 1, gt, 0.5, 0)
        frame2 = cv2.addWeighted(img, 1, pred, 0.5, 0)

        # cv2.imshow(f"GT", frame1)
        # cv2.putText(frame2, f"Loss BCE + Dice: {loss_mixed:.3f}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame2, f"Loss BCE:        {loss_bce:.3f}", (50, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame2, f"Loss Dice:       {loss_dice:.3f}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        # cv2.putText(frame2, f"Loss Jaccard:    {loss_jaccard:.3f}", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        def get_bg_color(value):
            # value in [0,1], threshold at 0.7
            if value >= 0.7:
                g = int(50 + 127 * min((value - 0.7) / 0.3, 1.0))
                return (0, g, 0)
            else:
                r = int(128 + 127 * min((0.7 - value) / 0.7, 1.0))
                return (0, 0, r)

        # Draw rectangles with color based on precision and recall
        prec_color = get_bg_color(precision)
        rec_color = get_bg_color(recall)
        img_name, _ = dataset.img_gt_list[img_index]
        cv2.putText(frame2, f"Image: {img_name}", (50, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame2, (45, 38), (200, 60), prec_color, -1)
        cv2.rectangle(frame2, (45, 63), (200, 85), rec_color, -1)
        cv2.putText(frame2, f"Precision: {precision:.2f}", (50, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame2, f"Recall: {recall:.2f}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
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
