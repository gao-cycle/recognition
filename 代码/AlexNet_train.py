import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc("font",family='SimHei') # 中文字体
plt.rcParams['axes.unicode_minus']=False  # 用来正常显示负号
import time
import os

import numpy as np
from tqdm import tqdm

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt


# 忽略烦人的红色提示
import warnings
warnings.filterwarnings("ignore")
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

from torchvision import transforms

# 训练集图像预处理：缩放裁剪、图像增强、转 Tensor、归一化
train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])

# 测试集图像预处理-RCTN：缩放、裁剪、转 Tensor、归一化
test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

# 数据集文件夹路径
dataset_dir = 'datasets'
train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'Validation')
print('训练集路径', train_path)
print('测试集路径', test_path)
from torchvision import datasets

# 载入训练集
train_dataset = datasets.ImageFolder(train_path, train_transform)

# 载入测试集
test_dataset = datasets.ImageFolder(test_path, test_transform)
print('训练集图像数量', len(train_dataset))
print('类别个数', len(train_dataset.classes))
print('各类别名称', train_dataset.classes)
class_names = train_dataset.classes
n_class = len(class_names)
print(train_dataset.class_to_idx)
# 映射关系：索引号 到 类别
idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}
# 保存为本地的 npy 文件
np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)
from torch.utils.data import DataLoader
BATCH_SIZE = 32

# 训练集的数据加载器
train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=0
                         )

# 测试集的数据加载器
test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=0
                        )
from torchvision import models
import torch.optim as optim
model = models.resnet18(pretrained=True) # 载入预训练模型

# 修改全连接层，使得全连接层的输出与当前数据集类别数对应
# 新建的层默认 requires_grad=True
model.fc = nn.Linear(model.fc.in_features, n_class)
print(model.fc)
# 只微调训练最后一层全连接层的参数，其它层冻结
model = model.to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 训练轮次 Epoch
EPOCHS = 20

# 遍历每个 EPOCH
for epoch in tqdm(range(EPOCHS)):

    model.train()
    for images, labels in train_loader:  # 获取训练集的一个 batch，包含数据和标注
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # 前向预测，获得当前 batch 的预测结果
        loss = criterion(outputs, labels)  # 比较预测结果和标注，计算当前 batch 的交叉熵损失函数
        optimizer.zero_grad()
        loss.backward()  # 损失函数对神经网络权重反向传播求梯度
        optimizer.step()  # 优化更新神经网络权重
    model.eval()
    all = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in tqdm(test_loader):  # 获取测试集的一个 batch，包含数据和标注
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # 前向预测，获得当前 batch 的预测置信度
            _, preds = torch.max(outputs, 1)  # 获得最大置信度对应的类别，作为预测结果
            total += labels.size(0)
            correct += (preds == labels).sum()  # 预测正确样本个数
            all = all + 1
            print(all)
        print('测试集上的准确率为 {:.3f} %'.format(100 * correct / total))
torch.save(model, 'fruit30_pytorch_C1.pth')