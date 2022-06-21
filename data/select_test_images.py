import pandas as pd
import numpy as np
import subprocess

data_dir = "/cluster/project/jbuhmann/dental_imaging/data/all_patches/"
frame = pd.read_csv("/cluster/home/emanete/dental_imaging/data/reduced.csv")
image_arr = np.asarray(frame.iloc[:, 1])

for image_name in image_arr:
    src_path = data_dir + image_name + ".dcm"
    dst_path = "./test_images"
    subprocess.run('scp emanete@euler.ethz.ch:' + src_path + ' ' + dst_path)




