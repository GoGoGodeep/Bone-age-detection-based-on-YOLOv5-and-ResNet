import numpy as np
from boneCrop import rectImage
from boneScore import dataset, DataLoader, Classify, SCORE, arthrosis, calcBoneAge
import os

# 仅需更改下面两行
label_path = 'runs/detect/exp6/labels'
sex = 'girl'    # 规定性别为女性

# 读取labels文件夹中的所有文件，在对应的图片中进行剪切
for i in os.listdir(label_path):
    img_name = i[:-4] + '.jpg'
    if os.path.exists('runs/detect/' + img_name):
        print("图片剪切完成！图片标签制作完成！")
        break
    else:
        rectImage(img_name, label_path, i)
        print("图片剪切完成！图片标签制作完成！")

score = 0

path = 'runs/detect/' + img_name + '/'
for i in ['DIPFifth.txt', 'DIPFirst.txt', 'DIPThird.txt',
          'MCPFifth.txt', 'MCPFirst.txt', 'MCPThird.txt',
          'MIPFifth.txt', 'MIPThird.txt', 'PIPFifth.txt',
          'PIPFirst.txt', 'PIPThird.txt', 'Radius.txt', 'Ulna.txt']:
    data = dataset(path + i, path)
    dataloader = DataLoader(data)
    category = arthrosis[i[:-4]]    # 读取分类
    classify = Classify(category, dataloader)

    out = classify()
    out = np.argmax(out.data.cpu())     # 选取输出的tensor数组中最大元素测坐标，即为等级类别
    score += SCORE[sex][i[:-4]][out]    # 根据计分表计分

print("score:", score)    # 输出分数
age = calcBoneAge(score, sex)
print("age:", age)      # 输出年龄