# Interview-Question
## 面试题库
##面试题库

# Interview-Question
- 面试题库

##30、什么是梯度爆炸？
### 解析：

误差梯度是神经网络训练过程中计算的方向和数量，用于以正确的方向和合适的量更新网络权重。在深层网络或循环神经网络中，误差梯度可在更新中累积，变成非常大的梯度，然后导致网络权重的大幅更新，并因此使网络变得不稳定。在极端情况下，权重的值变得非常大，以至于溢出，导致 NaN 值。

网络层之间的梯度（值大于 1.0）重复相乘导致的指数级增长会产生梯度爆炸。

##31、梯度爆炸会引发什么问题？


在深度多层感知机网络中，梯度爆炸会引起网络不稳定， https://zhuanlan.zhihu.com/p/24780433

最好的结果是无法从训练数据中学习，而最坏的结果是出现无法再更新的 NaN 权重值。

梯度爆炸导致学习过程不稳定。—《深度学习》，2016。
-

-- 在循环神经网络中，梯度爆炸会导致网络不稳定，无法利用训练数据学习，最好的结果是网络无法学习长的输入序列数据。

**如何提升机器人回环检测能力？**


18、Faster-rcnn相关
   - RCNN系列模型的区别， Faster R-CNN网络做了哪些改进/优化
  - 项目中使用Faster rcnn，请问Faster rcnn的优势是什么，为什么在这个项目使用Faster rcnn
  - Faster-rcnn RPN的作用和原理，RPN怎么计算 box 的实际坐标
  - 原始图片中的RoI如何映射到到feature map？    https://zhuanlan.zhihu.com/p/24780433



## 基础信息
> 图片检测、识别、分割的区别 ---- 进神经网络时第一个不同是，识别可以resize但是检测对位置敏感不能resize（传统上说，现在可以resize，但是记住缩放信息，能够还原）
>> 手推梯度反向传播
>>> 介绍熟悉的NASNet网络
	- 常用的凸优化方法，介绍
## 2、CNN

- BP 算法手推反向传播，cnn怎么更新w值和反向传播， 手推CNN公式
>> BP神经网络的结构是怎样的？和其他神经网络的区别是什么？
  - CNN反向传播公式推导
  - CNN的平移不变性是什么？如何实现的？CNN网络中的不变性理解
  - CNN是深度深好还是长宽大好
  - 神经网络怎样进行参数初始化？
  - CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？
	
