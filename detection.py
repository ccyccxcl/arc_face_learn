from face_recogntion.detection_face import DetectionFace
import argparse
import logging
import os
from conf import *
import cv2


def logging_handle(logging_path):
    # 保证log定义名字不一样，否则会全部写在一个日志下
    logger_name = os.path.splitext(os.path.split(logging_path)[1])[0]
    # 日志定义
    logger_handle = logging.getLogger(logger_name)
    logger_handle.setLevel(level=logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s  - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(logging_path)
    file_handler.setFormatter(formatter)
    logger_handle.addHandler(file_handler)

    return logger_handle


logger_handle = logging_handle(LOG_PATH)
df = DetectionFace(DBFACE_PATH, FACE_ACTION, logger_handle)


def run(image_path):
    img = cv2.imread(image_path)
    bounding_box_list, predict_name_list, predict_score_list = df.update(img)
    if bounding_box_list:
        for box, name, score in zip(bounding_box_list, predict_name_list, predict_score_list):
            print(box, name, score)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255),
                          thickness=2)
        cv2.imwrite("show.jpg", img)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image_path", help="image_path")
    args = parser.parse_args()
    if args.image_path:
        image_path = args.image_path
        run(image_path)
        print('||| Completed')
    else:
        print('||| Without query parameters, exit.')
