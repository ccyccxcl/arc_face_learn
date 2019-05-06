import numpy as np
import os
import cv2
import vision.face_recogntion.arc_face.arc_face_function as arc_face
from vision.face_recogntion.arc_face import *


class DetectionFace(object):
    def __init__(self, compare_emb_path, logger_face):
        self.logger_face = logger_face
        self.action_arc_face()
        self.engine_arc_face()
        self.feature_dict_dbface = self.load_face_data(compare_emb_path)

    def action_arc_face(self):
        res = arc_face.asfActivation(APP_ID, SDK_KEY)
        if res == 0 or res == 90114:
            self.logger_face.info(f"[INFO] Success arc face action res : {res}")
        else:
            self.logger_face.info(f"[INFO] Failed arc face action res : {res}")

    def engine_arc_face(self, detectMode=0xFFFFFFFF, detectFaceOrientPriority=0x5, detectFaceScaleVal=16,
                        detectFaceMaxNum=50,
                        combinedMask=1 | 4 | 8 | 16 | 32):
        res = arc_face.asfInitEngine(detectMode, detectFaceOrientPriority, detectFaceScaleVal, detectFaceMaxNum,
                                     combinedMask)
        if res[0] == 0:
            self.logger_face.info(f"[INFO] Success arc face engine res : {res[0]}")
        else:
            self.logger_face.info(f"[INFO] Failed arc face engine res : {res[0]}")

    def load_face_data(self, dataset_path):
        # 存放特征库
        feature_dict = {}
        images = os.listdir(dataset_path)
        if images:
            for image in images:
                img = arc_face.ARCFaceImg()
                img.filePath = os.path.join(dataset_path, image)
                im = arc_face.loadImage(img)
                res_detect, faces = arc_face.asfDetectFaces(im, 0x201)
                if res_detect == 0:
                    res_feature = arc_face.asfFaceFeatureExtract(im, 0x201, arc_face.getSingleFaceInfo(faces, 0))
                    if res_feature[0] == 0:
                        person_id = str(img.filePath).split(os.path.sep)[-1].replace(".jpg", "")
                        feature_dict[person_id] = res_feature[1]
                    else:
                        self.logger_face.info(f"[INFO] Failed dbface feature face image_path : {image}")
                else:
                    self.logger_face.info(f"[INFO] Failed dbface detection face image_path : {image}")
        return feature_dict

    def get_face_feature(self, frame):
        feature_dict = {}
        img = arc_face.ARCFaceImg()
        img.filePath = ""
        img.data = frame
        im = arc_face.loadImage(img)
        res_detect, faces = arc_face.asfDetectFaces(im, 0x201)
        if res_detect == 0:
            for i in range(faces.faceNum):
                res_feature = arc_face.asfFaceFeatureExtract(im, 0x201, arc_face.getSingleFaceInfo(faces, i))
                if res_feature[0] == 0:
                    feature_dict[i] = (faces.faceRect[i], res_feature[1])
                else:
                    self.logger_face.info(f"[INFO] Failed frame feature face res_feature : {res_feature[0]}")
        return feature_dict

    def update(self, frame):
        bounding_box_list = []
        predict_name_list = []
        predict_score_list = []
        feature_dic_frame = self.get_face_feature(frame)
        if len(feature_dic_frame) > 0:
            for i in list(feature_dic_frame.keys()):
                bounding_box_list.append(((
                    feature_dic_frame[i][0].left, feature_dic_frame[i][0].top, feature_dic_frame[i][0].right,
                    feature_dic_frame[i][0].bottom)))
                if len(self.feature_dict_dbface) > 0:
                    score_list = []
                    name_list = []
                    for key in list(self.feature_dict_dbface.keys()):
                        res, score = arc_face.asfFaceFeatureCompare(self.feature_dict_dbface[key],
                                                                    feature_dic_frame[i][1])
                        score = round(score, 4)
                        if res == 0:
                            if score >= 0.0:
                                score_list.append(score)
                                name_list.append(key)
                    score_pred = max(score_list)
                    score_pred_index = score_list.index(score_pred)
                    name_pred = name_list[score_pred_index]
                    predict_name_list.append(name_pred)
                    predict_score_list.append(score_pred)
        return bounding_box_list, predict_name_list, predict_score_list
