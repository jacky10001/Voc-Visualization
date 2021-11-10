
#%%
import os
import glob
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image

SAVE_IM_PATH = "save_data/im"
SAVE_GT_PATH = "save_data/mask"
os.makedirs(SAVE_IM_PATH, exist_ok=True)
os.makedirs(SAVE_GT_PATH, exist_ok=True)


#%% iterater

IM_PATH = "../VOC2012/JPEGImages"
GT_PATH = "../VOC2012/SegmentationClass"

with open("../VOC2012/ImageSets/Segmentation/trainval.txt") as f:
    t = f.read().split('\n')[:-1]

for name in t:
    ## get file path
    im_path = os.path.join(IM_PATH,name+".jpg")
    gt_path = os.path.join(GT_PATH,name+".png")

    ## read data
    im = Image.open(im_path)
    gt = Image.open(gt_path)
    gt = np.array(gt)
    gt[gt == 255] = 0
    gt[gt != 15] = 0
    gt[gt == 15] = 255

    ## get person data
    if 255 in gt:
        im.save( os.path.join(SAVE_IM_PATH,name+".jpg") )

        gt = gt
        gt = Image.fromarray(gt, mode='L')
        gt.save( os.path.join(SAVE_GT_PATH,name+".png") )
print("finished.")
