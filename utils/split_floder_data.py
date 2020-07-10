import os
import shutil
import random

# 注意修改路劲，要么直接给绝对路劲，如果相对路劲注意进到utils目录下运行
origin_img_path = '../x32'
train_path = '../data/train'
valid_path = '../data/valid'
test_path = '../data/test'

train_val_scale = 0.9
train_scale = 0.9
img_formats = ['.png', '.jpg','.bmp']

for label_name in os.listdir(origin_img_path):
    if not os.path.exists(os.path.join(train_path, label_name)):
        os.makedirs(os.path.join(train_path, label_name))
    if not os.path.exists(os.path.join(valid_path, label_name)):
        os.makedirs(os.path.join(valid_path, label_name))
    if not os.path.exists(os.path.join(test_path, label_name)):
        os.makedirs(os.path.join(test_path, label_name))

    all_label_img = os.listdir(os.path.join(origin_img_path, label_name))
    images = [os.path.join(origin_img_path, label_name, x) for x in all_label_img if
              os.path.splitext(x)[-1].lower() in img_formats]
    num_img = len(images)
    random.shuffle(images)
    train_val_num = int(num_img * train_val_scale)
    print(train_val_num)
    train_num = int(train_val_num * train_scale)
    print(train_num)
    trainval_img_list = images[0:train_val_num]
    test_img_list = images[train_val_num:]
    train_img_list = trainval_img_list[0:train_num]
    val_img_list = trainval_img_list[train_num:]
    for img in train_img_list:
        base_name = os.path.basename(img)
        shutil.copy(img, os.path.join(os.path.join(train_path, label_name, base_name)))
    for img in test_img_list:
        base_name = os.path.basename(img)
        shutil.copy(img, os.path.join(os.path.join(test_path, label_name, base_name)))
    for img in val_img_list:
        base_name = os.path.basename(img)
        shutil.copy(img, os.path.join(os.path.join(valid_path, label_name, base_name)))
