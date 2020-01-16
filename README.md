# [比赛记录-01：2020-1-15]

## 1.比赛概述

比赛名称：CACL高校AI挑战联赛第一赛季-第二轮

比赛题目：Cifar-10分类问题，Cifar-10数据集部分数据集共9类。

技术栈：Python、Pytorch（请务必安装GPU版本）

库：Pytorch==1.3.0, opencv-python==4.1.2.30

参考文章：https://myrtle.ai/how-to-train-your-resnet-8-bag-of-tricks/

https://colab.research.google.com/github/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb

 

## 2.代码执行过程

建议使用colab进行训练验证，本文将表述两个方法一个为colab，一个是本地验证。

本地验证：将train、test文件夹放在根目录，然后运行train.py即可运行。

**重要说明：**神经网络具有不确定性我们不能保证每次结果都能达到92.97的准确率，参考顺序：csv结果文件>已有的模型文件>重新训练并验证。

### 2.1colab验证

打开colab网址，务必将数据集的train文件夹与test文件夹放在根目录，文件目录如下：

```
-qzg_cifar10-2.0
----test
--------airplane
--------········
----train
--------test_airplane
--------········
----core.py
----dawn_utils.py
----read_data.py
----torch_backend.py
----train.py
----train.txt
----test.txt
----get_result.py
----model.pth（注：92.97%的模型）
```

其中train.py是对模型进行训练，直接运行即可，请务必使用GPU版本的pytorch。

##  3.优点

采用9层ResNet网络。

#### 3.1  混合精度训练

​       虽然采用适当的混合精度训练并不困难人，但这会增加整体训练时长，而且我们发现其对最终准确率并没有显著影响，所以我们继续选择忽略。

#### 3.2  移动最大池层的位置

​       采用 ![img](https://github.com/Qzgfather/Qzgfather.github.io/blob/master/assets/img/res.png)的网络架构，相比于常规架构缩减训练时间，但是会减少准确率。

#### 3.3 标签平滑

​        标签平滑在分类改进问题当中的一种相当成熟的神经网络训练速度与泛化技巧。其实质，是将独热目标概率与交叉熵损失内的各个类标签进行均匀混合。这有助于稳定输出分布结果，并防止网络进行过度自信的预测（可能抵制后续练效果）。

#### 3.4  CELU 激活

​        利用平滑激活函数对流程进行优化，借此取代初始阶段的 ReLU 及其具有曲率的 delta 函数。这种作法也能够提升模型的推广能力，因为平滑函数是一种表达式较少的函数类——在整体平滑限制之下，我们能够借此还原出一套线性网络。

#### 3.5  幽灵批量规范与冻结批量规范比例

​        幽灵批量规范采用的批量大小为 512。我们对权重衰减做出严格控制。我们将其冻结为 1/4 这一恒定值，大致相当于它们在训练中的中位平均值。我们将 CELU 的α参数调整为补偿因子 4，并将批量规范方差的学习率与权重衰减分别设定为 42 与 1/42，那么则可将批量标准比例修正为 1。

#### 3.6  输入补丁白化

​        批量规范可以很好地控制各个通道的分布，但并不能解决通道与像素之间的协方差问题。利用批量规范的“白化”版本，使我们有望控制内部层的协方差。

#### 3.7  指数移动平均化

​       高学习率是快速训练的必要前提，因为其允许随机梯度下降能够在有限的时间内在参数空间中推进必要的距离。在另一方面，学习率还需要在训练结束时进行退火，以确保模型能够优化参数空间中的浮动与噪声区间。参数平均化方法使得训练能够以更高学习率推进，同时通过多次迭代求平均的方式尽可能缩小浮动与噪声范围。

## 4.网络结构：

![SVG](https://github.com/Qzgfather/Qzgfather.github.io/blob/master/assets/img/net.svg)





​        模型验证代码，我们将验证部分放在了colab上，因为本地进行验证太慢了，我们将提供验证脚本进行本地验证（本地的验证脚本为**get_result.py**），我们还是建议在colab上直接运行，我们使用函数传入图片，返回的是一个标签构成的csv文件，并没有和测试集一一对应，然后又单独写了一个脚本，将图片的文件名和标签形成最终的结果，虽然有点脱裤子放屁的感觉，但是主要是懒，最终结果是f_result.csv文件。

```python
'''
验证模块
对模型进行实例化，然后去验证输入图片的类别，直接调用get_result函数传入路径耐心等待即可，一定要耐心，因为我没时间优化了现在已经2020年1月1日01:33:03了。
'''
import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import numpy as np
from torch_backend import *
from dawn_utils import net, tsv
from PIL import Image

classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'horse', 'ship', 'truck']
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 把模型转为test模式
    def get_result(path):
      img = Image.open(path)  # 读取要预测的图片
      img = np.array(img)
      trans = transforms.Compose(
          [
              transforms.ToTensor(),
              transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
          ])
      img = trans(img)
      img = img.half().to(device).unsqueeze(0)
      output = model({'input': img})
      a = output['logits'].max(1)[1].cpu().numpy()[0]
      return classes[a]

    # get_result("./train/airplane/batch_1_num_29.jpg")

    f = open('result.csv', 'a')
    f2 = open('./test.txt', 'r')
    data = [x.split('\t')[0] for x in f2.readlines()]
    data = list(map(get_result, data))
    f.write(tr(data))  
    f.close()
    f2.close()
```

```python
'''
上面的导出我直接把列表写入csv了，我在这里再整理一下，形成最终的结果，为啥直接写入列表？因为懒
'''
file = open('result.csv', 'r')
f2 = open('./test.txt', 'r')
f3 = open('f_result.csv', 'a')# 最终结果
test_file = [x.split('\t')[0].split('/')[-1] for x in f2.readlines()]  #双重split最为致命
result = file.read()
result = result.replace('[','').replace(']','').replace('\'', '').split(',')
print(len(test_file))
print(len(result))
print('完美')

for i in range(len(test_file)):
  f3.write(test_file[i] + ',' +result[i] + '\n')

print('恭喜最终结果生成完成！！！')

```









