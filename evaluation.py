import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from constants import IMG_HEIGHT, IMG_WIDTH
import torch.nn.functional as F
import warnings

warnings.filterwarnings("error")

def auc_from_roc_curve(tpr, fpr):
    # Sort by increasing FPR
    sorted_indices = np.argsort(fpr)
    fpr = np.array(fpr)[sorted_indices]
    tpr = np.array(tpr)[sorted_indices]

    # Trapezoidal integration
    return np.trapz(tpr, fpr)

if __name__ == "__main__":

    exp_name = "shallowUNET_v4_2conv_B_Lmix_R_ep15"
    pred_path = f"results/{exp_name}"
    gt_path = r"data\labels"
    
    files_ = os.listdir(pred_path)
    thr_vec = np.arange(0, 1, step=0.05)
    n_files, n_thrs = len(files_), len(thr_vec)

    tp_vec = np.zeros(n_thrs)
    fp_vec = np.zeros_like(tp_vec)
    fn_vec = np.zeros_like(tp_vec)

    for j, img_name in enumerate(files_):
        print(f"File {j}")
        img_name = img_name.split(".")[0]

        try:
            pred = cv2.imread(f"{pred_path}/{img_name}.png", cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(f"{gt_path}/mask_{img_name.split("_")[0]}_road_{img_name.split("_")[1]}.png", cv2.IMREAD_GRAYSCALE)
        except:
            print("WARNING: {img_name} unable to read, will be skipped for validation")
            continue

        # Pad gt to be of IMG_HEIGHT, IMG_WIDTH, crop gt to be of IMG_HEIGHT, IMG_WIDTH
        if (gt is not None):
            gt *= 255
            gt = gt[:IMG_HEIGHT, :IMG_WIDTH]
            pad_height = max(0, IMG_HEIGHT - gt.shape[0])
            pad_width = max(0, IMG_WIDTH - gt.shape[1])
            gt = np.pad(gt, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
            
            # cv2.imshow("pred", pred)
            # cv2.imshow("gt", gt)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            # prepare for metrics
            pred = pred.astype(float)/255. # in a range of 0 to 1 indicating probs
            gt = gt.astype(float)/255.   # [0, 1]

            # compute metrics on pixel similarity
            for i, thr in enumerate(thr_vec):
                pred_road = np.zeros_like(pred)
                pred_road[pred >= thr] = 1.

                tp = np.sum(pred_road * gt)  # both are ones
                fp = np.sum((pred_road > 0.95) & (gt == 0))
                fn = np.sum((pred_road < 0.05) & (gt > 0.95))

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