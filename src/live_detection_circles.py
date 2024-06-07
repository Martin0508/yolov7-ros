
'''
Copyright (c) 2024 DB Systemtechnik GmbH - Perzeptionslabor (TT.TVP4)

file:			live_detection_circles.py
created on:		2024-06-04, 10:21:31
created by:		Martin Geisler

last modified:	2024-06-06, 08:51:24
modified by:	Martin Geisler

description:		

input:			
subscriber:		

output:			
publisher:		

HISTORY:
Date      	By		Comments
**********	****	*****************************************************
'''

#!/usr/bin/env python3

import sys
import os
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import torch
from collections import deque

# Füge das YOLOv7-Verzeichnis zum Python-Pfad hinzu
sys.path.append('/home/student2/line_cam/ImageDetection/YOLO/yolov7/det')

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device

class YOLOv7:
    def __init__(self, weights, conf_thresh, iou_thresh, device, view_img, imgsz, savejson, plotfps, sub_topic, save_ocr, project, save_img, pub_topic):
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device)
        self.model.eval()
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.view_img = view_img
        self.imgsz = imgsz
        self.savejson = savejson
        self.plotfps = plotfps
        self.sub_topic = sub_topic
        self.save_ocr = save_ocr
        self.project = project
        self.save_img = save_img
        self.pub_topic = pub_topic
        self.bridge = CvBridge()
        self.image_buffer = deque(maxlen=3)

        rospy.Subscriber(sub_topic, Image, self.callback)

    def callback(self, data):
        try:
            # Konvertiere ROS-Bildnachricht in ein OpenCV-Bild
            cv_image = self.bridge.imgmsg_to_cv2(data, "mono8")
            self.image_buffer.append(cv_image)

            # Wenn der Puffer die erforderliche Größe erreicht hat, verarbeite die Bilder
            if len(self.image_buffer) == 3:
                # Kombiniere die Bilder zu einem Bild
                combined_image = np.vstack(self.image_buffer)

                # Bild auf das Modell anpassen
                img = combined_image.copy()
                img = img[np.newaxis, np.newaxis, :, :]  # fügt Batch- und Kanal-Dimension hinzu
                img = torch.from_numpy(img).to(self.device)
                img = img.float() / 255.0  # Bild normalisieren

                # YOLO-Modell zur Objekterkennung verwenden
                pred = self.model(img, augment=False)
                pred = non_max_suppression(pred, self.conf_thresh, self.iou_thresh, classes=None, agnostic=False)

                # Ergebnisse auf dem Bild anzeigen
                for det in pred:  # für jede Erkennung
                    if len(det):
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], combined_image.shape).round()
                        for *xyxy, conf, cls in reversed(det):
                            x1, y1, x2, y2 = map(int, xyxy)
                            cv2.rectangle(combined_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(combined_image, f'{self.model.names[int(cls)]} {conf:.2f}', (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Zeige das kombinierte Bild mit den erkannten Objekten an
                if self.view_img:
                    cv2.imshow("YOLO Detection", combined_image)
                    cv2.waitKey(1)

        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: {0}".format(e))

if __name__ == '__main__':
    rospy.init_node("yolov7_node")
    ns = rospy.get_name() + "/"
    
    weights_path    = rospy.get_param(ns + "weights_path")
    img_size        = rospy.get_param(ns + "img_size")
    conf_thresh     = rospy.get_param(ns + "conf_thresh")
    iou_thresh      = rospy.get_param(ns + "iou_thresh")
    device          = rospy.get_param(ns + "device")
    view_img        = rospy.get_param(ns + "view_img")
    save_ocr        = rospy.get_param(ns + "save_ocr")
    sub_topic       = rospy.get_param(ns + "sub_topic")
    plotfps         = rospy.get_param(ns + "plotfps")
    savejson        = rospy.get_param(ns + "save_json")
    project         = rospy.get_param(ns + "project")
    save_img        = rospy.get_param(ns + "save_img")
    pub_topic       = rospy.get_param(ns + "pub_topic")

    detect = YOLOv7(
        weights=weights_path,
        conf_thresh=conf_thresh,
        iou_thresh=iou_thresh,
        device=device,
        view_img=view_img,
        imgsz=img_size,
        savejson=savejson,
        plotfps=plotfps,
        sub_topic=sub_topic,
        save_ocr=save_ocr,
        project=project,
        save_img=save_img,
        pub_topic=pub_topic   
    )

    rospy.spin()
