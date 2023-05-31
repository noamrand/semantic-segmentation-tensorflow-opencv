import os
import random

import cv2
import numpy as np

cfg_path = "./models/mask_rcnn_inception/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt"
weights_path = "./models/mask_rcnn_inception/frozen_inference_graph.pb"
class_names_path = "./models/mask_rcnn_inception/mscoco_labels.names"

img_path = "./cat.png"


def get_detections(net, blob):
    net.setInput(blob)
    boxes, masks = net.forward(["detection_out_final", "detection_masks"])
    return boxes, masks



img = cv2.imread(img_path)
H, W, C = img.shape

net = cv2.dnn.readNetFromTensorflow(weights_path, cfg_path)

blob = cv2.dnn.blobFromImage(img)


boxes, masks = get_detections(net, blob)

empty_img = np.zeros(((H,W,C)))

detection_thres = 0.5


for j in range(len(masks)):
    bbox = boxes [0, 0, j]

    class_id = bbox[1]
    score = bbox[2]

    if score > detection_thres:
        mask = masks[j]
        x1,y1,x2,y2 = int(bbox[3]*W), int(bbox[4]*H), int(bbox[5]*W), int(bbox[6]*H)
        mask = mask[int(class_id)]

        mask = cv2.resize(mask, (x2-x1, y2-y1))
        _, mask = cv2.threshold(mask, 0.5, 1, cv2.THRESH_BINARY)
        mask = mask*255

        for c in range(3):
            empty_img[y1:y2, x1:x2, c] = mask

    
overlay = ((0.6 * empty_img) + (0.4 * img)).astype("uint8")

cv2.imshow("image", img)    
cv2.imshow("mask", empty_img)
cv2.imshow("overlay", overlay)
cv2.waitKey(0)
