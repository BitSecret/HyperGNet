import os
import shutil


source = "F:/PythonProject/InterGPS/data/geometry3k/train/"
target = "F:/PythonProject/geo3k_trans_data/"


image_dir = os.listdir(target)
for i in image_dir:
    if i.endswith(".png"):
        source_file = source + i.split("-")[1].split(".")[0] + "/img_diagram_point.png"
        target_file = target + "new/" + i
        shutil.copyfile(source_file, target_file)
