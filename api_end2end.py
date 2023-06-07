import math
import os
import cv2
import numpy as np
import argparse
import time
import random

path_Silent = './Silent-Face-Anti-Spoofing'
path_light = './light-weight-face-anti-spoofing'
import sys
sys.path.append(path_Silent)
sys.path.append(path_light)

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

import utils
from demo_tools import TorchCNN


def equalize(img): #input bgr output = BGR

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    img_hsv[:, :, 0] = cv2.equalizeHist(img_hsv[:, :, 0])
    image = cv2.cvtColor(img_hsv, cv2.COLOR_YCrCb2BGR)

    return image

def test_model1(image_ori, model_dir, device_id):
    # predict model Silent FAS

    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()
    image = image_ori
    image_bbox_list =model_test.get_bbox(image)

    prediction_list = []
    img = image
    for i in range(len(image_bbox_list)):
        image_bbox = image_bbox_list[i]
        prediction = 0
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

            img = image_cropper.crop(**param)
            if scale == 1.2:
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))[0][1] * 1.5
            else:
                prediction += model_test.predict(img, os.path.join(model_dir, model_name))[0][1]

        prediction_list.append(prediction/(len(os.listdir(model_dir)) + 0.5))

    return image_bbox_list, prediction_list, img

def predict_model2(image_face, spoof_model):

    output = spoof_model.forward(image_face)
    output = list(map(lambda x: x.reshape(-1), output))

    return output

def test_end2end(image, model_dir, device_id, spoof_model):
    # use multi model 
    image_bbox_list, prediction_list, face_img = test_model1(image, model_dir, device_id)
    if len(image_bbox_list) > 0:

        image_bbox_list = image_bbox_list[0]
        image_bbox = image_bbox_list
        prediction_model1 = prediction_list[0]
        prediction_model1 = [prediction_model1, 0 ]

        # padding box face in origin image
        pading_w = int(image_bbox[3] * 0.15)
        pading_h = int(image_bbox[2] * 0.15)
        box_final = [image_bbox[1] - pading_w , image_bbox[1] + image_bbox[3] + pading_w , image_bbox[0] -pading_h, image_bbox[0] + image_bbox[2] + pading_h]
        if pading_w > image_bbox[1]:
            box_final[0] = 0
        if pading_h > image_bbox[0]:
            box_final[2] = 0
        if image_bbox[1] + image_bbox[3] + pading_w > image.shape[0]:
            box_final[1] = image.shape[0]
        if image_bbox[0] + image_bbox[2] + pading_h > image.shape[1]:
            box_final[3] = image.shape[1]
 
        image_face = image[box_final[0] :box_final[1] , box_final[2] : box_final[3]]

        prediction_model2 = predict_model2([image_face], spoof_model)[0]
        prediction_model2 = [prediction_model2[1], prediction_model2[0]]

        predict_output = np.array(prediction_model1) * (5/9) + np.array(prediction_model2) * (4/9)
        value = predict_output[0]

        if  value > 0.52: # thresh_hold
            result_text = "Score: {:.2f}, {:.2f}".format(prediction_model1[0], prediction_model2[0])
            label = 0
            color = (255, 0, 0)

        else:
            label = 1
            result_text = "Score: {:.2f}, {:.2f}".format(prediction_model1[0], prediction_model2[0])
            color = (0, 0, 255)

        # draw box and score to image_origin
        cv2.rectangle(
            image,
            (image_bbox[0], image_bbox[1]),
            (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]),
            color, 3)
        
        cv2.putText(
            image,
            result_text,
            (image_bbox[0]-5, image_bbox[1] - 5),
            cv2.FONT_HERSHEY_COMPLEX, 0.4 * image.shape[0]/512, color)
        return label, image, image_face 
    else:
        label = None
        return label, image, image

if __name__ == "__main__":

    mydict = {"video":"./demo/KVID0011.mp4",
            "cam_id":0,
            "config":"./light-weight-face-anti-spoofing/configs/config.py",
            "spoof_thresh":0.7,
            "spf_model":"./CP/model_light/train_MN3_antispoof_2004_200k.pth.tar",
            "device":"cuda",
            "GPU":"0",
            "write_video":"True"
            }

    from argparse import Namespace
    args_model2 = Namespace(**mydict)

    config = utils.read_py_config(args_model2.config)
    print("config:::" ,config)
    spoof_model = utils.build_model(config, args_model2, strict=False, mode='eval')
    spoof_model = TorchCNN(spoof_model, args_model2.spf_model, config, device='cuda')

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
        help="video used to test")
    parser.add_argument(
        "--path_save_video",
        type=str,
        default= "./demo/output_video_demo.mp4",
        help="path_save_video")
    parser.add_argument(
        "--show_video",
        action='store_true',
        help="show_video_output")
    
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
    
    count = 0
    check = 0
    while True: 
        success, img = cap.read()  # guarantee first frame
        if success:
            count += 1
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            # img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
            t0 = time.time()
            label, image_output, img_face = test_end2end(img, args.model_dir, args.device_id, spoof_model)
            if label == 0:
                check += 1
            image_output = cv2.resize(image_output, resolution)
            writer_video.write(image_output)    
            if count % 30 == 0:
                print(f'frame_num:{count} on {frames}___',"time per img:", time.time() - t0)

            # if want show video output 
            if args.show_video :
                cv2.imshow('video_output', cv2.resize(image_output, (360, int(resolution[1] *360 / resolution[0])) ))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break

    print("check:", check/count)
    cap.release()
    writer_video.release()
    