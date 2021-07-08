import os
import cv2
import  xml.dom.minidom
 
image_path      = "../VOC2012/JPEGImages/"
annotation_path = "../VOC2012/Annotations/"

save_image_folder = "non_person"

os.makedirs(save_image_folder, exist_ok=True)


cnt = 0


files_name = os.listdir(image_path)
for filename_ in files_name:
    filename, extension= os.path.splitext(filename_)
    img_path =image_path+filename+'.jpg'
    xml_path =annotation_path+filename+'.xml'
    img = cv2.imread(img_path)
    if img is None:
        pass
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    objects=dom.getElementsByTagName("object")
    for i, object in enumerate(objects):
        cls_name = root.getElementsByTagName('name')[i].childNodes[0].data
        if cls_name != "person":
            bndbox = root.getElementsByTagName('bndbox')[i]
            xmin = bndbox.getElementsByTagName('xmin')[0]
            ymin = bndbox.getElementsByTagName('ymin')[0]
            xmax = bndbox.getElementsByTagName('xmax')[0]
            ymax = bndbox.getElementsByTagName('ymax')[0]

            # get data   str --> int
            xmin_data = int(xmin.childNodes[0].data)
            ymin_data = int(ymin.childNodes[0].data)
            xmax_data = int(xmax.childNodes[0].data)
            ymax_data = int(ymax.childNodes[0].data)

            roi_w = xmax_data - xmin_data
            roi_h = ymax_data - ymin_data
            if (abs(roi_w - roi_h) < 30) and (roi_w > 50 and roi_h > 50):
                roi = img[ymin_data:ymax_data, xmin_data:xmax_data]
                
                save_path = os.path.join(save_image_folder, cls_name)
                os.makedirs(save_path, exist_ok=True)
                save_path = save_path+"/%06d_%s.jpg"%(cnt, filename)
                # save_path = save_image_folder+"/%06d_%s.jpg"%(cnt, filename)
                flag = cv2.imwrite(save_path, roi)
                if(flag):
                    print(img_path, "      %2d"%len(objects), "      ", save_path)
                    cnt += 1

print("Crop ROI finished.")