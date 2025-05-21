import re
import numpy as np

def compute_tp_fp_fn(pred_road: np.ndarray, gt: np.ndarray):
    tp = np.sum(pred_road * gt)  # both are ones
    fp = np.sum((pred_road > 0.95) & (gt == 0))
    fn = np.sum((pred_road < 0.05) & (gt > 0.95))
    return tp, fp, fn


def pad_gt(gt: np.ndarray, h: int, w: int) -> np.ndarray:
    gt = gt[:h, :w]
    pad_height = max(0, h - gt.shape[0])
    pad_width = max(0, w - gt.shape[1])
    gt = np.pad(gt, ((0, pad_height), (0, pad_width)), mode='constant', constant_values=0)
    return gt


def extract_log_data(log_file_path):
    epochs = []
    train_losses = []
    validation_losses = []
    with open(log_file_path, 'r') as file:
        for line in file:
            match = re.search(r"Global epoch: (\d+) -> Train loss: ([\d.]+) \| Validation loss: ([\d.]+)", line)
            if match:
                epochs.append(int(match.group(1)))
                train_losses.append(float(match.group(2)))
                validation_losses.append(float(match.group(3)))
    return epochs, train_losses, validation_losses