import os
from PIL import Image

# 筛选需要的的关节信息
def bone(path, name):

    bonelist = []
    # 读取txt中的所有信息并加入列表
    with open(os.path.join(path, name), 'r') as f:
        line = f.readlines()
        for i in line:
            i = i.strip().split()
            bonelist.append(i)
    f.close()

    MiddlePhalanx = []      # 0: MiddlePhalanx
    DistalPhalanx = []      # 1: DistalPhalanx
    ProximalPhalanx = []    # 2: ProximalPhalanx
    MCP = []                # 3: MCP
    MCPFirst = []           # 4: MCPFirst
    Ulna = []               # 5: Ulna
    Radius = []             # 6: Radius

    # 根据类别分别加入列表
    # yolo检测出来的关节数必须为21
    if len(bonelist) == 21:
        for i in range(len(bonelist)):
            if bonelist[i][0] == '0':
                MiddlePhalanx.append(bonelist[i])
            elif bonelist[i][0] == '1':
                DistalPhalanx.append(bonelist[i])
            elif bonelist[i][0] == '2':
                ProximalPhalanx.append(bonelist[i])
            elif bonelist[i][0] == '3':
                MCP.append(bonelist[i])
            elif bonelist[i][0] == '4':
                MCPFirst.append(bonelist[i])
            elif bonelist[i][0] == '5':
                Ulna.append(bonelist[i])
            elif bonelist[i][0] == '6':
                Radius.append(bonelist[i])
    else:
        print("检测到的关节数目出错")
        exit()

    # 从小到大进行排列
    MiddlePhalanx = sorted(MiddlePhalanx)
    DistalPhalanx = sorted(DistalPhalanx)
    ProximalPhalanx = sorted(ProximalPhalanx)
    MCP = sorted(MCP)
    # 因为下面都只有一种，不需要进行排列
    # MCPFirst = sorted(MCPFirst)
    # Ulna = sorted(Ulna)
    # Radius = sorted(Radius)

    # 筛选出第一、三、五根手指
    DistalPhalanx.pop(1)    # 删除第二根手指
    DistalPhalanx.pop(2)    # 上面删除后变成四根手指，再删除原来的第四根
    MiddlePhalanx.pop(1)
    MiddlePhalanx.pop(2)
    ProximalPhalanx.pop(1)
    ProximalPhalanx.pop(2)
    MCP.pop(1)
    MCP.pop(2)

    # 返回不同关节的列表
    return DistalPhalanx, MiddlePhalanx, ProximalPhalanx, MCP, MCPFirst, Ulna, Radius

# 将yolov5输出的中心点和宽高转化成X1、Y1、X2、Y2
def coordinate(box):
    x1 = float(box[1]) - float(box[3]) / 2
    y1 = float(box[2]) - float(box[4]) / 2
    x2 = float(box[1]) + float(box[3]) / 2
    y2 = float(box[2]) + float(box[4]) / 2

    return x1, y1, x2, y2

# 剪切图片并保存
def cropImg(box, img, cls, img_name):
    for i in range(len(box)):
        # 因为yolov5输出的都是偏移量，需要乘以实际图片的大小
        x1, y1, x2, y2 = coordinate(box[i])
        x1 = x1 * img.size[0]
        x2 = x2 * img.size[0]
        y1 = y1 * img.size[1]
        y2 = y2 * img.size[1]

        img_ = img.crop((x1, y1, x2, y2))

        if not os.path.exists('runs/detect/' + img_name):
            os.makedirs('runs/detect/' + img_name)
        img_.save('runs/detect/' + img_name + '/' + \
                    cls + '_' + str(i) + '.png')
        if "Dis" in cls:
            if i == 0:
                file = open('runs/detect/' + img_name + '/' + "DIPFirst.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
            elif i == 1:
                file = open('runs/detect/' + img_name + '/' + "DIPThird.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
            else:
                file = open('runs/detect/' + img_name + '/' + "DIPFifth.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
        if 'MCPF' in cls:
            file = open('runs/detect/' + img_name + '/' + "MCPFirst.txt", 'a')
            file.write(cls + '_' + str(i) + '.png')
            file.close()
        if 'MCP' in cls and 'MCPF' not in cls:
            if i == 0:
                file = open('runs/detect/' + img_name + '/' + "MCPThird.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
            elif i == 1:
                file = open('runs/detect/' + img_name + '/' + "MCPFifth.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
        if 'Mid' in cls:
            if i == 0:
                file = open('runs/detect/' + img_name + '/' + "MIPThird.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
            elif i == 1:
                file = open('runs/detect/' + img_name + '/' + "MIPFifth.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
        if 'Pro' in cls:
            if i == 0:
                file = open('runs/detect/' + img_name + '/' + "PIPFirst.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
            elif i == 1:
                file = open('runs/detect/' + img_name + '/' + "PIPThird.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
            elif i == 2:
                file = open('runs/detect/' + img_name + '/' + "PIPFifth.txt", 'a')
                file.write(cls + '_' + str(i) + '.png')
                file.close()
        if 'Rad' in cls:
            file = open('runs/detect/' + img_name + '/' + "Radius.txt", 'a')
            file.write(cls + '_' + str(i) + '.png')
            file.close()
        if 'Ul' in cls:
            file = open('runs/detect/' + img_name + '/' + "Ulna.txt", 'a')
            file.write(cls + '_' + str(i) + '.png')
            file.close()

# 汇总
def rectImage(img_name, label_path, name):
    img_path = 'data/images/'
    img_path = os.path.join(img_path, img_name)

    img = Image.open(img_path)
    img = img.convert('RGB')
    # print(img.size)

    DistalPhalanx, MiddlePhalanx, ProximalPhalanx, \
        MCP, MCPFirst, Ulna, Radius = bone(label_path, name)

    # 根据最下面两个关节的位置
    # 左右手判断，右手就舍去
    if float(Radius[0][1]) > float(Ulna[0][1]):
        cropImg(DistalPhalanx, img, 'DistalPhalanx', img_name)
        cropImg(MiddlePhalanx, img, 'MiddlePhalanx', img_name)
        cropImg(ProximalPhalanx, img, 'ProximalPhalanx', img_name)
        cropImg(MCP, img, 'MCP', img_name)
        cropImg(MCPFirst, img, 'MCPFirst', img_name)
        cropImg(Ulna, img, 'Ulna', img_name)
        cropImg(Radius, img, 'Radius', img_name)
    else:
        print("为左手")
