import os
import numpy as np
from PIL import Image
import cv2 as cv
from tqdm import tqdm
import random
import shutil
Image.MAX_IMAGE_PIXELS = 1000000000000000
TARGET_W, TARGET_H = 1024, 1024


def cut_images(image_name, image_path, label_path, save_dir, is_show=True):
    # 初始化路径
    image_save_dir = os.path.join(save_dir, "images/"+image_name.split(".")[0])
    if not os.path.exists(image_save_dir): os.makedirs(image_save_dir)
    label_save_dir = os.path.join(save_dir, "labels/"+image_name.split(".")[0])
    if not os.path.exists(label_save_dir): os.makedirs(label_save_dir)
    if is_show:
        label_show_save_dir = os.path.join(save_dir, "labels_show/"+image_name.split(".")[0])
        if not os.path.exists(label_show_save_dir): os.makedirs(label_show_save_dir)
    
    target_w, target_h = TARGET_W, TARGET_H
    overlap = target_h // 8 # 128
    stride = target_h - overlap # 896
    
    image = np.asarray(Image.open(image_path))
    label = np.asarray(Image.open(label_path))
    image = cv.cvtColor(image,cv.COLOR_RGB2BGR)

    h, w = image.shape[0], image.shape[1]
    print("原始大小: ", w, h)
    if (w-target_w) % stride:
        new_w = ((w-target_w)//stride + 1)*stride + target_w
    if (h-target_h) % stride:
        new_h = ((h-target_h)//stride + 1)*stride + target_h
    image = cv.copyMakeBorder(image,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,value=[0,0,0])
    label = cv.copyMakeBorder(label,0,new_h-h,0,new_w-w,cv.BORDER_CONSTANT,value=1)
    h, w = image.shape[0], image.shape[1]
    print("填充至整数倍: ", w, h)

    def crop(cnt, crop_image, crop_label, is_show=is_show):
        _name = image_name.split(".")[0]
        image_save_path = os.path.join(image_save_dir, _name+"_"+str(cnt[0])+"_"+str(cnt[1])+".png")
        label_save_path = os.path.join(label_save_dir, _name+"_"+str(cnt[0])+"_"+str(cnt[1])+".png")
        label_show_save_path = os.path.join(label_show_save_dir, _name+"_"+str(cnt[0])+"_"+str(cnt[1])+".png")
        cv.imwrite(image_save_path, crop_image)
        cv.imwrite(label_save_path, crop_label)
        if is_show:
            cv.imwrite(label_show_save_path, crop_label*255)
    
    h, w = image.shape[0], image.shape[1]
    cnt = 0
    for i in tqdm(range((w-target_w)//stride + 1)):
        for j in range((h-target_h)//stride + 1):
            topleft_x = i*stride
            topleft_y = j*stride
            crop_image = image[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
            crop_label = label[topleft_y:topleft_y+target_h,topleft_x:topleft_x+target_w]
            if np.sum(crop_image) != 0:
                crop((i, j),crop_image,crop_label)
                cnt += 1
    print(cnt)
    # os.remove(image_path)


def get_train_val():
    file_train = open('./data/train.txt', 'w')
    file_val = open('./data/val.txt', 'w')

    image_list_382 = os.listdir('./data/images/382')
    image_list_182 = os.listdir('./data/images/182')

    print(len(image_list_182))
    print(len(image_list_382))

    random.shuffle(image_list_382)
    random.shuffle(image_list_182)

    for ele in image_list_382:
        if random.randint(0, 10) < 2:  # 8:2
            file_val.write(str(ele) + '\n')
        else:
            file_train.write(str(ele) + '\n')

    for ele in image_list_182:
        if random.randint(0, 10) < 2:  # 8:2
            file_val.write(str(ele) + '\n')
        else:
            file_train.write(str(ele) + '\n')

    file_train.close()
    file_val.close()


if __name__ == "__main__":
    data_dir = "./data"
    img_name1 = "382.png"
    img_name2 = "182.png"
    label_name1 = "382_label.png"
    label_name2 = "182_label.png"
    cut_images(img_name1, os.path.join(data_dir, img_name1), os.path.join(data_dir, label_name1), data_dir)
    cut_images(img_name2, os.path.join(data_dir, img_name2), os.path.join(data_dir, label_name2), data_dir)
    get_train_val()
