import argparse
import inspect
import os.path as osp
import sys

path_light = './light-weight-face-anti-spoofing'
sys.path.append(path_light)
import cv2
import glog as log
import numpy as np
# from IPython import display

# current_dir = osp.dirname(osp.abspath(inspect.getfile(inspect.currentframe())))
# parent_dir = osp.dirname(current_dir)
# sys.path.insert(0, parent_dir)
# print("parent_dir", parent_dir)
# exit()

import utils
from demo_tools import TorchCNN, VectorCNN, FaceDetector

from retinaface.pre_trained_models import get_model
import time

def pred_spoof(frame, detections, spoof_model):
    """Get prediction for all detected faces on the frame"""
    faces = []
    for rect in detections:
        if len(rect["bbox"])==0:
            continue
        left, top, right, bottom =  (int(tx) for tx in rect["bbox"])
        # cut face according coordinates of detections
        # if (top-bottom)*(left - right) != 0:
        face = frame[top:bottom, left:right]
        h, w = face.shape[0], face.shape[1]
        if h*w != 0:
            faces.append(face)
    if len(faces) > 0:
        # print("faces",faces)
        output = spoof_model.forward(faces)
        output = list(map(lambda x: x.reshape(-1), output))
        return output
    return None

def draw_detections(frame, detections, confidence, thresh):
    """Draws detections and labels"""
    for i, rect in enumerate(detections):
        if len(rect["bbox"])==0:
            continue
        left, top, right, bottom = (int(tx) for tx in rect["bbox"])
        if confidence[i][1] > thresh:
            label = f'spoof: {round(confidence[i][1]*100, 3)}%'
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), thickness=2)
        else:
            label = f'real: {round(confidence[i][0]*100, 3)}%'
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), thickness=2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)
        top = max(top, label_size[1])
        cv2.rectangle(frame, (left, top - label_size[1]), (left + label_size[0], top + base_line),
                     (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))

    return frame

def run(params, capture, face_det, spoof_model, write_video=False):
    """Starts the anti spoofing demo"""
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    resolution = (640,480)
    fps = capture.get(cv2.CAP_PROP_FPS)  
    writer_video = cv2.VideoWriter('output_video_demo.mp4', fourcc, fps, resolution)
    win_name = 'Antispoofing Recognition'
    while True:
        has_frame, frame = capture.read()
        if has_frame:
            # print("frame", frame.shape)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            t0 = time.time()

            detections = face_det.predict_jsons(frame)
            print("time======", time.time() - t0)

            print("detections" , detections)
            confidence = pred_spoof(frame, detections, spoof_model)
            # print("confidence", confidence)
            if confidence != None:
                frame = draw_detections(frame, detections, confidence, params.spoof_thresh)
            output = cv2.resize(frame, resolution)
            # print(output.shape, "output")
            cv2.imshow(win_name, output)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # if write_video:
            #     writer_video.write(output)
 
    # capture.release()
    writer_video.release()
    cv2.destroyAllWindows()

def main():
    #load model face detection
    model = get_model("resnet50_2020-07-20", max_size=128)
    model.eval()

    ################## Params #################
    mydict = {"video":"//media/anlab/0e731fe3-5959-4d40-8958-e9f6296b38cb/home/anlab/songuyen/Download/data_spoof/archive/cut-out printouts/0001e96803--6239b2990123a46de8bb6937.mp4",
            "cam_id":0,
            "config":"/home/anlab/Desktop/Songuyen/Face Anti-spoofing/antispoofing/light-weight-face-anti-spoofing/configs/config.py",
            "spoof_thresh":0.7,
            "spf_model":"/home/anlab/Desktop/Songuyen/Face Anti-spoofing/antispoofing/spf_models/spf_models/MN3_antispoof.pth.tar",
            "device":"cuda",
            "GPU":"0",
            "write_video":"True"
            }
            
    from argparse import Namespace
    args = Namespace(**mydict)
    ############################################        

    device = args.device + f':{args.GPU}' if args.device == 'cuda' else 'cpu'
    write_video = args.write_video

    if args.cam_id >= 0:
        log.info('Reading from cam {}'.format(args.cam_id))
      
        cap = cv2.VideoCapture(args.cam_id)

        # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    else:
        assert args.video
        log.info('Reading from {}'.format(args.video))
        cap = cv2.VideoCapture(args.video)
        # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    # assert cap.isOpened()
    #face_detector = FaceDetector(args.fd_model, args.fd_thresh, args.device, args.cpu_extension)
    if args.spf_model.endswith('pth.tar'):
        if not args.config:
            raise ValueError('You should pass config file to work with a Pytorch model')
        config = utils.read_py_config(args.config)
        spoof_model = utils.build_model(config, args, strict=True, mode='eval')
        spoof_model = TorchCNN(spoof_model, args.spf_model, config, device=device)
    else:
        assert args.spf_model.endswith('.xml')
        spoof_model = VectorCNN(args.spf_model)
    # running demo
    run(args, cap, model, spoof_model, write_video)

if __name__ == '__main__':
    main()
