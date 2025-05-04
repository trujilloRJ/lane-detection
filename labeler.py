import numpy as np
import cv2
import os

KEY_ESC = 27
KEY_SPACE = 32
KEY_A = 97
KEY_D = 100
KEY_N = 110
KEY_M = 109
KEY_G = 103
KEY_S = 115

sel_x = None
sel_y = None

def click_callback(event, x, y, *args):
    global sel_x, sel_y
    sel_x, sel_y = x, y
    # if event == cv2.EVENT_LBUTTONDOWN:


if __name__=="__main__":

    img_folder = r"C:\javier\personal_projects\computer_vision\data\KITTI_road_segmentation\data_road\testing\image_2"
    gt_out_folder = r"data/testing"
    files_ = os.listdir(img_folder)

    cv2.namedWindow("Labeler")
    cv2.setMouseCallback("Labeler", click_callback)

    run = True
    img_index = 0
    is_loaded = False
    draw_margin = 10
    while(run):
        if not is_loaded:
            img_name = files_[img_index]
            img = cv2.imread(f"{img_folder}/{img_name}")
            mask = np.zeros_like(img)
            is_loaded = True

        frame = cv2.addWeighted(img, 1, mask, 0.5, 0)
        cv2.putText(frame, f"Margin: {draw_margin}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Labeler", frame)

        key = cv2.waitKey(0)

        if (sel_x is not None) & (sel_y is not None):
            mask[sel_y - draw_margin : sel_y + draw_margin, sel_x - draw_margin : sel_x + draw_margin, 1] = 254

        if key == KEY_D:
            img_index += 1
            is_loaded = False
        if key == KEY_A:
            img_index -= 1
            is_loaded = False
        if key == KEY_M:
            draw_margin = min(draw_margin + 10, 100)
        if key == KEY_N:
            draw_margin = max(draw_margin - 10, 1)
        if key == KEY_SPACE:
            cv2.imwrite(f"{gt_out_folder}/{img_name}", mask)
        if key == KEY_ESC:
            run = False
        else:
            print(key)

    cv2.destroyAllWindows()
