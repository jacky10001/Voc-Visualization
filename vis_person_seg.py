
#%%
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
from PIL import Image

gt = Image.open("../VOC2012/SegmentationClass/2007_000799.png")
gt = np.array(gt)
gt[gt == 255] = 0
print(gt.max(), gt.min())
print(gt.shape)
print(gt[350,200])


gt[gt != 15] = 0
plt.imshow(gt)
plt.show()
