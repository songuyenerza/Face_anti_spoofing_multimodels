# -*- coding: utf-8 -*-
# @Time : 20-6-9 下午3:06
# @Author : zhuying
# @Company : Minivision
# @File : test.py
# @Software : PyCharm
path_Silent = './Silent-Face-Anti-Spoofing'
import sys
sys.path.append(path_Silent)
from email.mime import image
import math

import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


# 因为安卓端APK获取的视频流宽高比为3:4,为了与之一致，所以将宽高比限制为3:4
def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        # print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def detect_live(image_name, model_dir, device_id):
    model_detect = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = image_name
 
    image_bbox_list = [model_detect.get_bbox(image)]
    for i in range(len(image_bbox_list)):
        image_bbox = image_bbox_list[i]
        prediction = np.zeros((1, 3))
        test_speed = 0
        # sum the prediction from single model's result
        for model_name in os.listdir(model_dir):
            h_input, w_input, model_type, scale = parse_model_name(model_name)
            param = {
                "org_img": image,
                "bbox": image_bbox,
                "scale": scale,
                "out_w": w_input,
                "out_h": h_input,
                "crop": True,
            }

            if scale is None:
                param["crop"] = False
            # print(param)
            img = image_cropper.crop(**param)

            start = time.time()
            prediction += model_detect.predict(img, os.path.join(model_dir, model_name))
            test_speed += time.time()-start

        # draw result of prediction
        label = np.argmax(prediction)
        value = prediction[0][label]/2
        if label == 1 and value > 0.7 :
            result_text = "RealFace Score: {:.2f}".format(value)
            color = (255, 0, 0)
        else:
            result_text = "FakeFace Score: {:.2f}".format(value)
            color = (0, 0, 255)
        print("Prediction cost {:.2f} s".format(test_speed))
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 2)
        cv2.putText(
            image,
            result_text,
            (image_bbox[0], image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.5*image.shape[0]/1024, color)

    return image

if __name__ == "__main__":
    desc = "test"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--device_id",
        type=int,
        default=0,
        help="which gpu id, [0/1/2/3]")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="./CP/model_Silent",
        help="model_Silent used to test")
    parser.add_argument(
        "--path_video",
        type=str,
        default= 0,
        help="image used to test")
    parser.add_argument(
        "--path_save_video",
        type=str,
        default= "output_video_demo.mp4",
        help="path_save_video")
    
    args = parser.parse_args()

    s = args.path_video
    cap = cv2.VideoCapture(s)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("w, h of video::", w, h)
    
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    resolution = (h, w)
    fps  = int(cap.get(cv2.CAP_PROP_FPS))
    writer_video = cv2.VideoWriter(args.path_save_video, fourcc, fps, resolution)
    fps = cap.get(cv2.CAP_PROP_FPS)  # warning: may return 0 or nan
    frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0) or float('inf')  # infinite stream fallback
    fps = max((fps if math.isfinite(fps) else 0) % 100, 0) or 30  # 30 FPS fallback
    while True:
        success, img = cap.read()  # guarantee first frame
        if success:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            t0 = time.time()

            image_output = detect_live(img, args.model_dir, args.device_id)
            image_output = cv2.resize(image_output, resolution)

            writer_video.write(image_output)
            # print("time per img:", time.time() - t0)
            # if want show video
            # cv2.imshow('frame', image_output)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        writer_video.release()
    

