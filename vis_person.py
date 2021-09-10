import os
import cv2
import  xml.dom.minidom
 
image_path      = "../VOC2012/JPEGImages/"
annotation_path = "../VOC2012/Annotations/"

save_image_path = "Visualization_Person"

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
    root = dom.documentElement
    objects=dom.getElementsByTagName("object")
    for i, object in enumerate(objects):
        cls_name = root.getElementsByTagName('name')[i].childNodes[0].data
        print(".........................................               ", cls_name)
        if cls_name in ["person"]:
            bndbox = root.getElementsByTagName('bndbox')[i]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            ymin = bndbox.getElementsByTagName('ymin')[0]
            xmax = bndbox.getElementsByTagName('xmax')[0]
            ymax = bndbox.getElementsByTagName('ymax')[0]
            xmin_data=xmin.childNodes[0].data
            ymin_data=ymin.childNodes[0].data
            xmax_data=xmax.childNodes[0].data
            ymax_data=ymax.childNodes[0].data
            img = cv2.rectangle(img,(int(xmin_data),int(ymin_data)),(int(xmax_data),int(ymax_data)),(55,255,155),5)
    flag=0
    flag=cv2.imwrite(save_image_path+"/{}.jpg".format(filename),img)
    if(flag):
        print(filename,"done", "      %d"%len(objects))

print("all done ====================================")