import random
import cv2
import os
from PIL import Image

# 自适应直方图均衡化增强用于yolo的图片
def clahe_yolo():
    Images_dir = r"E:\OneDrive\DATA\BoneAge_Detect\datasets\Images"
    for i in os.listdir(Images_dir):
        img = cv2.imread(os.path.join(Images_dir, i), 0)

        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        dst = clahe.apply(img)

        cv2.imwrite(os.path.join(Images_dir, i), dst)
        # cv2.imshow("dst", dst)
        # cv2.waitKey(0)
        # cv2.destroyAllwindows()


# 自适应直方图均衡化增强用于分类的图片
def clahe_cls():
    image_dir = r'E:\OneDrive\DATA\BoneAge_Detect\detect'

    for tag in os.listdir(image_dir):
        num_dir = os.path.join(image_dir, tag)
        for num in os.listdir(num_dir):
            img_dir = os.path.join(num_dir, num)
            # print(img_dir)
            for img_name in os.listdir(img_dir):
                img = cv2.imread(os.path.join(img_dir, img_name), 0)

                # 自适应直方图均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
                dst = clahe.apply(img)

                save_dir = r'E:\OneDrive\DATA\BoneAge_Detect\detect'
                save_dir = os.path.join(save_dir, tag)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_dir = os.path.join(save_dir, num)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cv2.imwrite(os.path.join(save_dir, img_name), dst)
        exit()

# 将图像进行（-45°, 45°）翻转进行数据增样
def img_rotate():
    img_path = r'E:\OneDrive\DATA\BoneAge_Detect\detect'
    save_path = r'E:\OneDrive\DATA\BoneAge_Detect\detect_'
    flag = 5

    for tag in os.listdir(img_path):
        num_dir = os.path.join(img_path, tag)
        save_num_dir = os.path.join(save_path, tag)
        for num in os.listdir(num_dir):
            img_dir = os.path.join(num_dir, num)
            save_img_dir = os.path.join(save_num_dir, num)
            for img_name in os.listdir(img_dir):
                print('*', end='')
                img = Image.open(os.path.join(img_dir, img_name))
                save_img_name = os.path.join(save_img_dir, img_name)
                for i in range(flag):  # 每张图像翻转五次生成五张
                    rota = random.randint(-45, 45)
                    dst = img.rotate(rota)
                    dst.save(save_img_name[:-4] + '_' + str(i) + '.png')
        print(tag, "完成")


# clahe_cls()
# img_rotate()