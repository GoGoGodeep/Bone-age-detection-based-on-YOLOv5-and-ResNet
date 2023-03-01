import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torch import nn, optim
from PIL import Image
import os
import numpy as np

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
device = "cuda"

# ------------------------------------数据集处理------------------------------------
# 根据训练种类更改路径
label_file_dir = r"D:\OneDrive\DATA\BoneAge_Detect\detect\Ulna\labels.txt"
img_base_dir = r"D:\OneDrive\DATA\BoneAge_Detect\detect\Ulna\images"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])  # 对数据进行处理

# 9个大类，每个大类里面有多少个等级分类
# 方便循环训练9个分类模型
arthrosis = {'MCPFirst': ['MCPFirst', 11],  # 第一手指掌骨
             'DIPFirst': ['DIPFirst', 11],  # 第一手指远节指骨
             'PIPFirst': ['PIPFirst', 12],  # 第一手指近节指骨
             'MIP': ['MIP', 12],  # 中节指骨（除了拇指剩下四只手指）（第一手指【拇指】是没有中节指骨的））
             'Radius': ['Radius', 14],  # 桡骨
             'Ulna': ['Ulna', 12],  # 尺骨
             'PIP': ['PIP', 12],  # 近节指骨（除了拇指剩下四只手指）
             'DIP': ['DIP', 11],  # 远节指骨（除了拇指剩下四只手指）
             'MCP': ['MCP', 10]}  # 掌骨（除了拇指剩下四只手指）

# 数据集处理
class dataset(Dataset):
    def __init__(self):
        with open(label_file_dir) as f:
            self.dataset = f.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        line = self.dataset[index]
        strs = line.split()

        # —————————————————————————————————图片处理—————————————————————————————————
        _img_data = Image.open(os.path.join(img_base_dir, strs[0]))  # 打开图片数据

        # 等比缩放
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

        # —————————————————————————————————标签处理—————————————————————————————————
        label = np.array(list(map(float, strs[1:])))

        return bg_img, label

# 训练器
class Train(nn.Module):
    def __init__(self, category, dataloader, cls_name):
        super().__init__()

        # 直接调用resnet34模型并重写输入（输入为黑白图像）和输出
        self.net = models.resnet34(weights=models.ResNet34_Weights.DEFAULT).to(device)
        self.net.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False).to(device)
        self.net.fc = nn.Linear(512, category[1]).to(device)  # 根据类别数不同，分类数也不同

        self.loss = nn.CrossEntropyLoss()   # 多分类任务，使用交叉熵损失
        self.opt = optim.Adam(self.net.parameters())

        self.dataloader = dataloader
        self.cls_name = cls_name

    def __call__(self):

        try:
            # 读取模型参数
            self.net.load_state_dict(torch.load("boneClsCkpt/{0}.pth".format(self.cls_name)))
            print("加载成功")
        except:
            pass

        for epoch in range(60):
            print("轮次：", epoch)

            for i, (img, tag) in enumerate(self.dataloader):
                self.net.train()

                img = img.to(device)
                tag = tag.squeeze(1).to(device)

                out = self.net.forward(img)
                loss = self.loss(out, tag.long())
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            print("loss:", loss.item())

            if (epoch + 1) % 10 == 0:
                # 根据类别数不同保存相应的模型参数
                torch.save(self.net.state_dict(), "boneClsCkpt/{0}.pth".format(self.cls_name))
                print("参数保存成功")

if __name__ == '__main__':

    data = dataset()
    dataloader = DataLoader(dataset=data, shuffle=True, batch_size=128)
    # 根据训练种类更改
    category = arthrosis['Ulna']
    train = Train(category, dataloader, 'Ulna')
    train()