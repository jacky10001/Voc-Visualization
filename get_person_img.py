import os
import cv2
import  xml.dom.minidom
 
image_path      = "../VOC2012/JPEGImages/"
annotation_path = "../VOC2012/Annotations/"

save_image_path = "Visualization_Person"

os.makedirs(save_image_path, exist_ok=True)


obj_list= [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "hand",
    "head",
    "horse",
    "motorbike",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

 
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
    root = dom.documentElement
    objects=dom.getElementsByTagName("object")
    flag = 0
    for i, object in enumerate(objects):
        cls_name = root.getElementsByTagName('name')[i].childNodes[0].data
        if cls_name not in obj_list:
            print(".........................................               ", cls_name)
            flag = 1
    if flag:
        cv2.imwrite(save_image_path+"/{}.jpg".format(filename),img)
        print(filename,"done", "      %d"%len(objects))

print("all done ====================================")