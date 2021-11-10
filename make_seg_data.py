
#%%
import os
import glob
import numpy as np
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from PIL import Image

SAVE_IM_PATH = "save/im"
SAVE_GT_PATH = "save/mask"
os.makedirs(SAVE_IM_PATH, exist_ok=True)
os.makedirs(SAVE_GT_PATH, exist_ok=True)


#%% iterater
IM_PATH = "../VOC2012/JPEGImages"
GT_PATH = "../VOC2012/SegmentationClass"

with open("../VOC2012/ImageSets/Segmentation/trainval.txt") as f:
    t = f.read().split('\n')[:-1]

PAD_W = 500 ; PAD_H = 500

cnt = 0
for name in t:    
    ## get file path
    im_path = os.path.join(IM_PATH,name+".jpg")
    gt_path = os.path.join(GT_PATH,name+".png")

    ## read data
    im = io.imread(im_path)
    gt = Image.open(gt_path)
    gt = np.array(gt)
    gt[gt == 255] = 0
    gt[gt != 15] = 0
    gt[gt == 15] = 255

    ## get person data
    if 255 in gt:
        pa_im = np.zeros((PAD_H, PAD_W, 3), dtype="uint8")
        pa_gt = np.zeros((PAD_H, PAD_W), dtype="uint8")
        
        rsh = PAD_H/im.shape[0]
        rsw = PAD_W/im.shape[1]
        rs = rsh if rsh < rsw else rsw
        
        im = ( (im-im.min()) / (im.max()-im.min()) )
        im = transform.resize(im, (int(im.shape[0]*rs),int(im.shape[1]*rs))) * 255.
        im = im.astype("uint8")
        
        gt = ( (gt-gt.min()) / (gt.max()-gt.min()) )
        gt = transform.resize(gt, (int(gt.shape[0]*rs),int(gt.shape[1]*rs))) * 255.
        gt = gt.astype("uint8")
        
        pw1 = (PAD_W - im.shape[1])//2
        pw2 = im.shape[1] + pw1
        ph1 = (PAD_H - im.shape[0])//2
        ph2 = im.shape[0] + ph1
        
        pa_im[ph1:ph2,pw1:pw2] = im
        pa_gt[ph1:ph2,pw1:pw2] = gt
        
        io.imsave("%s/%06d_im.jpg"%(SAVE_IM_PATH, cnt), pa_im)
        io.imsave("%s/%06d_lb.jpg"%(SAVE_GT_PATH, cnt), pa_gt)
        cnt += 1

print("finished.")