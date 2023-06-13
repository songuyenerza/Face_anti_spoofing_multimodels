# Face Anti Spoofing
Towards the solving anti-spoofing problem on RGB only data.
Inference model to mobile app for real time testing.
## Introduction
Todo...
## Setup
* Python 3.6.15
* Torch 1.9
* scikit-image 0.17.2 
* opencv-python 4.1.2.30 
* Albumentations 1.0.0 
* Pillow 8.4.0
### Installation

1. Create a virtual environment:
```bash
cd light-weight-face-anti-spoofing/
bash init_venv.sh
```

2. Activate the virtual environment:
```bash
. venv/bin/activate
```
## Run inference
Run on video (multi_models)
```bash
python api_end2end.py
```
Todo: add logic with multi frames ...
## Tuning two model
1. Model MobileNet (Original_code :[github](https://github.com/kprokofi/light-weight-face-anti-spoofing.git)):
- Edit path folder images and path at file train_turning.py. Path '.csv' have format 0: spoof, 1: live
```bash
cd light-weight-face-anti-spoofing/
create path: python creat_path.py
training: python train_tuning.py  --config=./configs/config.py
```
2. Model FASNet (Original_code: [github](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git)):
- Install env from Original_code: [github](https://github.com/minivision-ai/Silent-Face-Anti-Spoofing.git)
```bash
cd Silent-Face-Anti-Spoofing/
edit at file config ./src/default_config.py
python train_tuning.py
```
## Convert model to format Onnx
Todo...

## APK source code  
Open source for Android platform deployment code:...
