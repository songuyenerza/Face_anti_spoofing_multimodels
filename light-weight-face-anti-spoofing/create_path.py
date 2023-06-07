import os

import numpy as np
import pandas as pd
import random

# csv_file = open(os.path.join(input_dir, csv_filename), "w")
data_folder = "/media/anlabadmin/data_ubuntu/SoNg/spoofing/data_350k/data_train_200k/spoofing/1_80x80/"
csv_file = open( 'train_path.csv', "w")
csv_file.write("fname,liveness_score\n")
lisst_ = []
# for filename, label in zip(fname, y):
for clas in os.listdir(data_folder):
    path_clas = os.path.join(data_folder, clas)
    # label = int(clas)
    for path in os.listdir(path_clas):
        path_img = os.path.join(path_clas, path)
        name = os.path.join(clas, path)
        lisst_.append(name)
random.shuffle(lisst_)
for name in lisst_:
    label = str(name.split('/')[0])
    if label == "1" or label == "live" or label == "live_2":
        label_int = 1
    else:
        label_int = 0

    csv_file.write(f"{name},{label_int}\n")
csv_file.close()
