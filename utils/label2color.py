import numpy as np

def label_img_to_color(img): #bgr
    label_to_color = {
        0: [0,0,0],
        1: [255, 255,255]
        }
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3),dtype=np.uint8)
    for cls in range(2):
        img_color[img==cls] = np.array(label_to_color[cls])
    return img_color

def diff_label_img_to_color(img): #bgr
    label_to_color = {
        255:[128,128,128],
        0: [0,0,0],
        1: [255, 255,255]
        }
    img_height, img_width = img.shape
    img_color = np.zeros((img_height, img_width, 3),dtype=np.uint8)
    for cls in [0,1,255]:
        img_color[img==cls] = np.array(label_to_color[cls])
    return img_color