import os

train_base_dir = r"E:\OneDrive\DATA\BoneAge_Detect\detect"
images_base_dir = r"E:\OneDrive\DATA\BoneAge_Detect\detect\Ulna\images"
labels_dir = r"E:\OneDrive\DATA\BoneAge_Detect\detect\Ulna\labels.txt"

# 图片重命名
def changeName(path):
    list = os.listdir(path)
    for i in range(1, len(list)):
        list_path = os.path.join(path, list[i])
        list_name =os.listdir(list_path)
        for name in list_name:
            i = 0
            for img in os.listdir(os.path.join(list_path, name)):
                dir = list_path + "/" + name + '/'
                os.rename(dir + img, dir + name + "_" + str(i) + ".png")
                i += 1


# 标签制作
def MKLabel(img_path, label_path):
    list = os.listdir(img_path)
    for name in list:
        with open(label_path, 'a') as la:
            if name[0:2] == '10':
                la.write(name + '\t' + str(9) + '\n')
            elif name[0:2] == '11':
                la.write(name + '\t' + str(10) + '\n')
            elif name[0:2] == '12':
                la.write(name + '\t' + str(11) + '\n')
            elif name[0:2] == '13':
                la.write(name + '\t' + str(12) + '\n')
            elif name[0:2] == '14':
                la.write(name + '\t' + str(13) + '\n')
            else:
                la.write(name+'\t' + str(int(name[0])-1)+'\n')
        la.close()


# changeName(train_base_dir)
MKLabel(images_base_dir, labels_dir)