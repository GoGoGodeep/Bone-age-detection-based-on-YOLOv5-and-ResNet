import math
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torch import nn
from PIL import Image
import os
import numpy as np

device = 'cuda'
transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])  # 对数据进行处理

# 根据性别和关节位置来进行打分
SCORE = {'girl': {
    'Radius': [10,15,22,25,40,59,91,125,138,178,192,199,203, 210],
    'Ulna': [27,31,36,50,73,95,120,157,168,176,182,189],
    'MCPFirst': [5,7,10,16,23,28,34,41,47,53,66],
    'MCPThird': [3,5,6,9,14,21,32,40,47,51],
    'MCPFifth': [4,5,7,10,15,22,33,43,47,51],
    'PIPFirst': [6,7,8,11,17,26,32,38,45,53,60,67],
    'PIPThird': [3,5,7,9,15,20,25,29,35,41,46,51],
    'PIPFifth': [4,5,7,11,18,21,25,29,34,40,45,50],
    'MIPThird': [4,5,7,10,16,21,25,29,35,43,46,51],
    'MIPFifth': [3,5,7,12,19,23,27,32,35,39,43,49],
    'DIPFirst': [5,6,8,10,20,31,38,44,45,52,67],
    'DIPThird': [3,5,7,10,16,24,30,33,36,39,49],
    'DIPFifth': [5,6,7,11,18,25,29,33,35,39,49]
},
    'boy':{
    'Radius': [8,11,15,18,31,46,76,118,135,171,188,197,201,209],
    'Ulna': [25,30,35,43,61,80,116,157,168,180,187,194],
    'MCPFirst': [4,5,8,16,22,26,34,39,45,52,66],
    'MCPThird': [3,4,5,8,13,19,30,38,44,51],
    'MCPFifth': [3,4,6,9,14,19,31,41,46,50],
    'PIPFirst': [4,5,7,11,17,23,29,36,44,52,59,66],
    'PIPThird': [3,4,5,8,14,19,23,28,34,40,45,50],
    'PIPFifth': [3,4,6,10,16,19,24,28,33,40,44,50],
    'MIPThird': [3,4,5,9,14,18,23,28,35,42,45,50],
    'MIPFifth': [3,4,6,11,17,21,26,31,36,40,43,49],
    'DIPFirst': [4,5,6,9,19,28,36,43,46,51,67],
    'DIPThird': [3,4,5,9,15,23,29,33,37,40,49],
    'DIPFifth': [3,4,6,11,17,23,29,32,36,40,49]
    }
}

# 13个关节对应的分类模型
arthrosis ={
    'MCPFirst': ['MCPFirst', 11],
            'MCPThird': ['MCP', 10],
            'MCPFifth': ['MCP', 10],

            'DIPFirst': ['DIPFirst', 11],
            'DIPThird': ['DIP', 11],
            'DIPFifth': ['DIP', 11],

            'PIPFirst': ['PIPFirst', 12],
            'PIPThird': ['PIP', 12],
            'PIPFifth': ['PIP', 12],

            'MIPThird': ['MIP', 12],
            'MIPFifth': ['MIP', 12],

            'Radius': ['Radius', 14],
            'Ulna': ['Ulna', 12]
}

# 输入性别和分数计算对应的年龄
def calcBoneAge(score, sex):
    if sex == 'boy':
        boneAge = 2.01790023656577 + (-0.0931820870747269)*score + math.pow(score, 2) * 0.00334709095418796 +\
        math.pow(score, 3) * (-3.32988302362153E-05) + math.pow(score, 4) * (1.75712910819776E-07) +\
        math.pow(score, 5) * (-5.59998691223273E-10) + math.pow(score, 6) * (1.1296711294933E-12) +\
        math.pow(score, 7) * (-1.45218037113138e-15) + math.pow(score, 8) * (1.15333377080353e-18) +\
        math.pow(score, 9) * (-5.15887481551927e-22) + math.pow(score, 10) * (9.94098428102335e-26)

        return round(boneAge, 2)
    elif sex == 'girl':
        boneAge = 5.81191794824917 + (-0.271546561737745)*score + \
        math.pow(score, 2) * 0.00526301486340724 + math.pow(score, 3) * (-4.37797717401925E-05) +\
        math.pow(score, 4) * (2.0858722025667E-07) + math.pow(score, 5) * (-6.21879866563429E-10) + \
        math.pow(score, 6) * (1.19909931745368E-12) + math.pow(score, 7) * (-1.49462900826936E-15) +\
        math.pow(score, 8) * (1.162435538672E-18) + math.pow(score, 9) * (-5.12713017846218E-22) +\
        math.pow(score, 10) * (9.78989966891478E-26)

        return round(boneAge, 2)

# 读取各个关节的txt文件制作对应数据集
class dataset(Dataset):
    def __init__(self, dir, path):
        with open(dir, 'r') as f:
            self.dataset = f.readlines()
        self.path = path

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        line = self.dataset[index]
        strs = line.strip().split()

        # —————————————————————————————————图片处理—————————————————————————————————
        _img_data = Image.open(os.path.join(self.path, strs[0]))  # 打开图片数据

        # 等比缩放，否则推理的时候会报错
        bg_img = torchvision.transforms.ToPILImage()(torch.zeros(1, 224, 224))

        img_size = torch.Tensor(_img_data.size)
        # 获取最大边长的索引
        l_max_index = img_size.argmax()
        ratio = 224 / img_size[l_max_index]
        img_resize = img_size * ratio
        img_resize = img_resize.long()

        img_use = _img_data.resize(img_resize)

        bg_img.paste(img_use)
        bg_img = transforms(bg_img)

        label = ''      # 推理不需要label

        return bg_img, label

# 进行推理
class Classify(nn.Module):
    def __init__(self, category, dataloader):
        super().__init__()
        self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
        self.net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False).to(device)
        self.net.fc = nn.Linear(512, category[1]).to(device)
        self.dataloader = dataloader
        self.category = category

    def __call__(self):
        # 测试
        self.net.eval()

        for i, (img, _) in enumerate(self.dataloader):
            # 根据不同的关节加载不同的模型参数
            self.net.load_state_dict(torch.load("boneClsCkpt/{0}.pth".format(self.category[0])))
            print(self.category[0] + "加载成功!")
            img = img.to(device)
            out = self.net.forward(img)

            return out

if __name__ == '__main__':
    sex = 'boy'    # 规定性别为女性
    score = 0
    # 方便起见，这里固定图片数据，也可以传变量
    path = 'runs/detect/1999.png/'
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