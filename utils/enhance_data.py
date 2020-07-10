from PIL import ImageEnhance
import os
import cv2
import numpy as np
from PIL import Image
import glob
import random

# 注意修改路劲
imageDir = "/opt/code/classify/photos/3"  # 要改变的图片的路径文件夹
saveDir = "/opt/code/classify/photos/3"  # 数据增强生成图片的路径文件夹
is_bright = 0.5
is_contrast = 0.5
is_rotation = 0.5


def brightnessEnhancement(root_path, img_name):  # 亮度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_bri = ImageEnhance.Brightness(image)
    # brightness = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    brightness = 1.5
    image_brightened = enh_bri.enhance(brightness)
    return image_brightened


def contrastEnhancement(root_path, img_name):  # 对比度增强
    image = Image.open(os.path.join(root_path, img_name))
    enh_con = ImageEnhance.Contrast(image)
    # contrast = 1.1+0.4*np.random.random()#取值范围1.1-1.5
    contrast = 1.5
    image_contrasted = enh_con.enhance(contrast)
    return image_contrasted


def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    random_angle = np.random.randint(-2, 2) * 90
    if random_angle == 0:
        rotation_img = img.rotate(-90)  # 旋转角度
    else:
        rotation_img = img.rotate(random_angle)  # 旋转角度
    return rotation_img


def flip(root_path, img_name):  # 翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return filp_img


for name in os.listdir(imageDir):
    print(name)
    base_name = os.path.basename(name)
    if random.random() < is_bright:
        saveName = "contrast_" + name
        saveImage = contrastEnhancement(imageDir, name)
        saveImage.save(os.path.join(saveDir, saveName))
    if random.random() < is_bright:
        saveName = "flip_" + name
        saveImage = flip(imageDir, name)
        saveImage.save(os.path.join(saveDir, saveName))
    if random.random() < is_bright:
        saveName = "bright_" + name
        saveImage = brightnessEnhancement(imageDir, name)
        saveImage.save(os.path.join(saveDir, saveName))
