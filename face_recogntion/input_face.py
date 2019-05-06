import numpy as np
import os
import cv2
import vision.face_recogntion.arc_face.arc_face_function as arc_face
from vision.face_recogntion.arc_face import *


class InputFace(object):
    def __init__(self, logger_web):
        self.logger_web = logger_web
        self.action_arc_face()
        self.engine_arc_face()

    def action_arc_face(self):
        res = arc_face.asfActivation(APP_ID, SDK_KEY)
        if res == 0 or res == 90114:
            self.logger_web.info(f"[INFO] Success arc face action res : {res}")
        else:
            self.logger_web.info(f"[INFO] Failed arc face action res : {res}")

    def engine_arc_face(self, detectMode=0xFFFFFFFF, detectFaceOrientPriority=0x5, detectFaceScaleVal=16,
                        detectFaceMaxNum=50,
                        combinedMask=1 | 4 | 8 | 16 | 32):
        res = arc_face.asfInitEngine(detectMode, detectFaceOrientPriority, detectFaceScaleVal, detectFaceMaxNum,
                                     combinedMask)
        if res[0] == 0:
            self.logger_web.info(f"[INFO] Success arc face engine res : {res[0]}")
        else:
            self.logger_web.info(f"[INFO] Failed arc face engine res : {res[0]}")

    def detection_face_one(self, img_path):
        img = arc_face.ARCFaceImg()
        img.filePath = os.path.join(img_path)
        im = arc_face.loadImage(img)
        res_detect, faces = arc_face.asfDetectFaces(im, 0x201)
        if res_detect == 0:
            if faces.faceNum == 1:
                return True
            else:
                return False
        else:
            return False
