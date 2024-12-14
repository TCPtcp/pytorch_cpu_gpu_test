import torch
import torchvision
from torch import nn
# from model import *
from torch.nn import Conv2d, MaxPool2d, Flatten, Linear
import time  # 计时器
# 添加tensorboard
from torch.utils import tensorboard
from torch.utils.tensorboard import SummaryWriter


# 准备数据集-训练数据集 + 测试数据集
train_data = torchvision.datasets.CIFAR10("./dataset/", train=True, download=True,
                                          transform=torchvision.transforms.ToTensor())
test_data = torchvision.datasets.CIFAR10("./dataset/", train=False, download=True,
                                          transform=torchvision.transforms.ToTensor())

# 获取数据长度
train_data_size = len(train_data)
test_data_size  = len(test_data) 
print("训练数据集的长度为：{}".format(train_data_size))  # 字符串格式化写法
print("测试数据集的长度为：{}".format(test_data_size))  # 字符串格式化写法

# 利用DataLoader来加载数据集
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_data, batch_size=256*4)
test_dataloader  = DataLoader(test_data, batch_size=256*4)

# 创建网络模型
# 搭建神经网络
# 数据为10分类数据，所以我们要搭建一个10分类的网络
class Tudui(nn.Module):
    def __init__(self):  # 初始化
        super(Tudui, self).__init__()  # 父类初始化
        self.model = nn.Sequential(
            Conv2d(3, 32, 5, 1, 2),  # 输入通道数，输出通道数，卷积核大小，步长，填充层大小
            MaxPool2d(2),  # 2指的是kernel_size的大小
            Conv2d(32, 32, 5, 1, 2) ,
            MaxPool2d(2),
            Conv2d(32, 64, 5, 1, 2), 
            MaxPool2d(2), 
            Flatten(), 
            Linear(64*4*4, 64), 
            Linear(64, 10)
        )

    def forward(self, x):
        output = self.model(x)
        return output


# 实例化模型
tudui = Tudui()

device = torch.device("cpu")  # 默认使用CPU
# 判断GPU是否可用
if torch.cuda.is_available():
    device = torch.device("cuda")  #  .cuda()  将模型放到GPU上
# 判断是否为mac的pytorch
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")  

# 定义损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.to(device)  # 将损失函数放到GPU上

# 定义优化器
learning_rate = 0.01  # 1e-2
optimizer = torch.optim.SGD(tudui.parameters(), lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练轮数
epoch = 10




writer = SummaryWriter("./logs_train")

# 添加时间计时器
start_time = time.time()
# %env OMP_NUM_THREADS=10
for i in range(epoch):
    print("----------第 {} 轮训练开始----------".format(i+1))

    # 训练步骤开始
    # tudui.train()  # 训练模式
    for data in train_dataloader:
        imgs, targets = data
        imgs = imgs.to(device)  # 将数据放到GPU上
        targets = targets.to(device)  # 将数据放到GPU上
        outputs = tudui(imgs)
        loss = loss_fn(outputs, targets)
        
        # 优化器优化模型
        optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 损失反向传播
        optimizer.step()  # 对参数进行优化
        
        total_train_step += 1  # 记录训练次数
        if total_train_step % 100 == 0:  # 控制一下不要每次训练都打印训练损失（否则就太长了）
            end_time = time.time()
            print(end_time - start_time)
            print("训练次数：{}，Loss：{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)  # 
    # 测试步骤开始
    # tudui.eval()  # 测试模式
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():  # 测试的时候不需要计算梯度
        for data in test_dataloader:
            imgs, targets = data
            imgs = imgs.to(device)  # 将数据放到GPU上
            targets = targets.to(device)  # 将数据放到GPU上
            outputs = tudui(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss += loss
            accuracy = (outputs.argmax(1) == targets).sum()
            total_accuracy += accuracy
    print("整体测试集上的Loss：{}".format(total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy/test_data_size))
    writer.add_scalar("test_loss", total_test_loss, total_test_step)
    writer.add_scalar("test_accuracy", total_accuracy/test_data_size, total_test_step)
    total_test_step += 1

    torch.save(tudui, "tudui_{}".format(i+1))
    # torch.save(tudui().state_dict(), "tudui_{}.pth".format(i))
    print("模型已保存")
writer.close()

end_time = time.time()

use_time = end_time - start_time
print("总计用时为：{}".format(use_time))