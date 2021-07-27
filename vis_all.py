import os
import cv2
import xml.dom.minidom
import numpy as np
 
image_path="D:\\Datasets\\VOC\\VOCdevkit\\VOC2007\\JPEGImages/"
annotation_path="D:\\Datasets\\VOC\\VOCdevkit\\VOC2007\\Annotations/"

save_image_path = "Visualization"

os.makedirs(save_image_path, exist_ok=True)
 
files_name = os.listdir(image_path)
for filename_ in files_name:
    filename, extension= os.path.splitext(filename_)
    img_path =image_path+filename+'.jpg'
    xml_path =annotation_path+filename+'.xml'
    print(img_path)
    img = cv2.imread(img_path)
    if img is None:
        pass
    dom = xml.dom.minidom.parse(xml_path)
    # root = dom.documentElement
    objects = dom.getElementsByTagName("object")
    for i, object in enumerate(objects):
        cls_name = object.getElementsByTagName('name')[0].childNodes[0].data
        bndbox = object.getElementsByTagName('bndbox')[0]
        xmin = bndbox.getElementsByTagName('xmin')[0]
        ymin = bndbox.getElementsByTagName('ymin')[0]
        xmax = bndbox.getElementsByTagName('xmax')[0]
        ymax = bndbox.getElementsByTagName('ymax')[0]
        xmin_data = xmin.childNodes[0].data
        ymin_data = ymin.childNodes[0].data
        xmax_data = xmax.childNodes[0].data
        ymax_data = ymax.childNodes[0].data
        color = (int(np.random.randint(100,255,1)[0]), int(np.random.randint(100,255,1)[0]), int(np.random.randint(100,255,1)[0]))
        img = cv2.rectangle(img,(int(xmin_data),int(ymin_data)),(int(xmax_data),int(ymax_data)), color ,2)
        img = cv2.putText(img, cls_name, (int(xmin_data),int(ymin_data)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    flag=0
    flag=cv2.imwrite("Visualization/{}.jpg".format(filename),img)
    if(flag):
        print(filename,"done", "      %d"%len(objects))

print("all done ====================================")