import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from constants import IMG_HEIGHT, IMG_WIDTH
import torch.nn.functional as F
import warnings
import tqdm
from common import compute_tp_fp_fn, pad_gt

warnings.filterwarnings("error")

def auc_from_roc_curve(tpr, fpr):
    # Sort by increasing FPR
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]

    # Trapezoidal integration
    return np.trapz(tpr, fpr)

if __name__ == "__main__":

    exp_name = "sUNetWide_v8_Srop_adam"
    pred_path = f"results/{exp_name}"
    gt_path = r"data\labels"
    
    files_ = os.listdir(pred_path)
    thr_vec = np.arange(0, 1, step=0.01)
    n_files, n_thrs = len(files_), len(thr_vec)

    tp_vec = np.zeros(n_thrs)
    fp_vec = np.zeros_like(tp_vec)
    fn_vec = np.zeros_like(tp_vec)

    for j, img_name in enumerate(tqdm.tqdm(files_)):
        img_name = img_name.split(".")[0]

        split_ = img_name.split("_")
        if len(split_) == 1:
            continue

        mask_type = split_[0]
        img_index = split_[1]

        try:
            pred = cv2.imread(f"{pred_path}/{img_name}.png", cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(f"{gt_path}/mask_{mask_type}_road_{img_index}.png", cv2.IMREAD_GRAYSCALE)
        except:
            print("WARNING: {img_name} unable to read, will be skipped for validation")
            continue

        # Pad gt to be of IMG_HEIGHT, IMG_WIDTH, crop gt to be of IMG_HEIGHT, IMG_WIDTH
        if (gt is not None):
            gt = pad_gt(gt, IMG_HEIGHT, IMG_WIDTH)
            
            # cv2.imshow("pred", pred)
            # cv2.imshow("gt", gt)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # prepare for metrics
            pred = pred.astype(float)/255. # in a range of 0 to 1 indicating probs
            gt = gt.astype(float)          # [0, 1]

            # compute metrics on pixel similarity
            for i, thr in enumerate(thr_vec):
                pred_road = np.zeros_like(pred)
                pred_road[pred >= thr] = 1.

                tp, fp, fn = compute_tp_fp_fn(pred_road, gt)

                tp_vec[i] += tp
                fp_vec[i] += fp
                fn_vec[i] += fn

    precision = tp_vec/(tp_vec + fp_vec)
    recall = tp_vec/(tp_vec + fn_vec)
    f1_score = tp_vec/(tp_vec + 0.5*(fp_vec + fn_vec))

    # AUC = auc_from_roc_curve(recall_vec, raw_metrics[1, :])
    # print(AUC)

    fig = plt.figure(figsize=(12, 4))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[:, 1])

    ax1.minorticks_on()
    ax1.grid(which='minor', linestyle='--', alpha=0.5)
    ax1.grid(which='major', linestyle='-')
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.set_xlim(-0.05, 1.05)
    ax1.set_ylim(-0.05, 1.05)
    ax1.plot(recall, precision, '-')

    ax2.minorticks_on()
    ax2.grid(which='minor', linestyle='--', alpha=0.5)
    ax2.grid(which='major', linestyle='-')
    ax2.set_xlabel("Threshold")
    ax2.plot(thr_vec, precision, '-', label="precision")
    ax2.plot(thr_vec, recall, '-', label="recall")
    ax2.plot(thr_vec, f1_score, '-', label="F1 score")
    ax2.set_ylim(0, 1)
    ax2.legend()

    fig.suptitle(exp_name)
    fig.savefig(f"{pred_path}/metrics.png")
    plt.show()