
### 简介
这是一个学习Pytorch和神经网络的入门笔记，记录了在课程学习中的所有代码实践过程，课程地址[1]

### 内容介绍
- datasets: datasets文件夹中存放的是一个比较小的图片数据集CIFAR-10,其中训练集为5万张32x32的三通道彩色图片，测试集为1万张32x32的三通道彩色图片，图片的种类共有10种，更详细的介绍，参考官网 [2]
- logs: logs文件夹中存放的是用board函数产生的训练过程中loss曲线，用tensborad展示
- step： step文件夹中存放的是训练过程各轮次的训练结果
- CIFAR10_moedl_test.py: 用于测试训练后的模型准确率
- indexDataset.py: 课程中写的所有代码都存放在了这一个文件，因为第一节讲的是关于数据集的内容，所以命名为indexDataset

### 参考资料
[1] [【PyTorch深度学习快速入门教程（绝对通俗易懂！）【小土堆】】 ](https://www.bilibili.com/video/BV1hE411t7RN/?share_source=copy_web&vd_source=129025ad6ca937c62eddc193e8ee5efb)

[2] [The CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html)