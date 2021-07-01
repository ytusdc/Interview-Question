# 深度学习相关

## 1、CNN 相关 + 基本方法

  - BP 算法手推反向传播
  - CNN反向传播公式推导
  - CNN的平移不变性是什么？如何实现的？CNN网络中的不变性理解
  - CNN是深度深好还是长宽大好
  - 神经网络怎样进行参数初始化？
	
	
## 2、卷积相关

  - 1 * 1卷积核的作用，哪些情况可以使用1x1卷积？
  - 3×3 卷积核 与 5×5 卷积核相比的优点？
  - 感受野的计算，CNN 的感受野受什么影响
  - CNN 网络模型的计算力（flops）和参数量（parameters）：普通卷积、DW PW 卷积计算量推导
  - 实现卷积操作(代码) 
  - 卷积神经网络的卷积核大小、个数,卷积层数如何确定呢?
  - 卷积神经网络的卷积核为什么是方形？为什么是奇数
  - 空洞卷积及其优点
   > 串联会产生Griding Efect 网格效应（棋盘格效应）
  - deformable conv怎么做,具体怎么学的，对偏移有没有什么限制
  - 卷积如何加速
  - 卷积层如何修剪，量化等
  
  
  - 上采样、下采样还有哪些类型
     卷积是降采样过程
	- upsample 上采样方法(语义分割)
	
	1、up-sampling--（Billnear intrpolation双线性插值）， 临近插值
     > 介绍常见的插值算法
    2、Transpose Conv （反卷积）反卷积具体怎么实现的？
    3、Up-Pooling

	
## 3、Pool 池化的作用和种类

  - Max Pooling和 Average Pooling的区别，使用场景分别是什么？
  - 当出现一个大噪点时，用哪种池化更好
  - Pooling怎么反向传播
  - 除了池化层还有什么方法可以减少特征数量(UNet模型，是采样的方法) 
  - RoI Pooling和RoI Align区别,  RoI Pooling 哪些操作导致其精度没有RoI Align高	
  - RoI Pooling的计算过程，写一下推理过程

## 4、模型评估方法

  - 精确率，召回率，和准确度评价怎么算，这俩是矛盾的怎么选最优
  - Accuracy作为指标有哪些局限性？
  - ROC曲线和AUC曲线意义，ROC曲线上每个点代表的含义
	    AUC原理，为什么更适用于排序问题？  AUC怎么算的？ROC曲线怎么画的
  - AUC指标有什么特点？放缩结果对AUC是否有影响？
  - 介绍F1-score, auc比F1好在哪-- ROC曲线 vs Precision-Recall曲线，各自的使用场景选择
  - Log Loss 和 AUC 的区别， 适用于什么场景
  - 分类、检测、分割评价指标说一下？
  - 手写AUC曲面面积的计算(或者伪代码)
  - 混淆矩阵
  - AP和mAP的区别？
  - 余弦距离与欧式距离有什么特点？https://www.zhihu.com/question/19640394
  - 什么是偏差和方差


## 5、正则化方法 L1、L2

  - L0、L1、L2定义，L1，L2正则化原理，从参数分布和参数图上解释
  - L1，L2 norm的区别，L1为什么能使特征稀疏，L2为什么不能使特征稀疏，L2为什么能解决(或者减轻) 过拟合
   > (L1范数，使权重为0，对应的特征则不起作用，使特征稀疏稀疏矩阵) 
  - Lasso、线性回归、逻辑回归、l1 l2 正则有什么影响，
   > 理解：L1正则先验分布是Laplace分布，L2正则先验分布是Gaussian分布
  - 口述一下 L1参数分布的推导(牛皮) , L1在0处不可导，怎么处理:利用坐标轴下降法或者proximal operator  
     http://roachsinai.github.io/2016/08/03/1Proximal_Method/
  - L1是损失函数，有哪些优化方法，能用sgd么？为什么？
  - L1是不可导的，真的可以用么？

	
## 6、各种激活函数的优缺点

  - 激活函数有什么用？各种激活函数（sigmoid，tanh，ReLU, leaky ReLU, PReLU, ELU （ReLU变体））介绍一下，优缺点及适用场景
  - 深度学习用relu激励函数，为什么，好处是什么
  - 比如sigmod的问题在哪里，relu是怎么解决的，relu的问题在哪里，有没有对应的解决算法(prelu) ，介绍 Leaky Relu 并写公式 leakyrelu解决了梯度消失问题吗
  - relu激励函数训练数据时会让神经元失活，训练之后进行应用时有什么注意的吗，训练时失活的神经元在应用时怎么办
  - Relu不可导怎么办


## 7、深度学习常用的optimizer(优化器) 

  - 介绍一下你经常用的optimize
  - 手推梯度反向传播
  - 凸优化了解吗？牛顿法、SGD、最小二乘法，各自的优势。
  - SGD与牛顿法的区别在哪？
  - loss优化的几个方法(sgd、动量、adam) 
  - Adam优化器的迭代公式, AdaGrad梯度的原理
  - 动量法的表达式，随机梯度下降相比全局梯度下降好处是什么
  - SGD每步做什么，为什么能online learning
  - 你一般用哪个优化器，为什么用它？介绍你知道的优化器
  - 学习率调节策略	
  - 为什么动量可以走出局部最小值？数学解释
  - 训练时出现loss NAN的可能因素	
  - adam用到二阶矩的原理是什么
  - 各个优化器的参数怎么设置


## 8、损失函数(Loss) 

  - 常用的 Loss 函数 (MSE /huber loss/BCE/cross entropy/指数损失/smooth l1 /softmax loss )  
  - cross entropy的原理是什么？反向传播机制是什么？交叉熵的公式伪代码
  - 为什么分类用交叉熵而不用MSE(均方误差)，同为什么使用交叉熵作为损失函数去评估误差
  - 为什么损失函数用交叉熵
  - 多标签分类怎么解决，从损失函数角度考虑	
  - Loss不降、不收敛的原因和解决方法
  - 损失函数正则项的本质是什么?
  - 样本不均衡怎么搞(重点考核损失函数优化 focal loss) 
  - 分割网络的损失函数
	
	
## 9、Softmax 相关

  - Softmax的原理是？反向传播、梯度公式推导，代码实现， 手推softmax的BP公式
  - 为什么softmax是指数形式
  - LR + softmax做多分类区别, 为什么 LR 用交叉熵损失而不是平方损失？
  - softmax、多个logistic的各自的优势？
  - 介绍分层 softmax，还有负采样？怎么负采样？
  - softmax 怎么防止溢出， softmax减去最大数字不变的证明
  - softmax得出的结果是排序的么，为什么分类用交叉熵
  - One-hot有什么作用？
  - 交叉熵和最大似然估计的关系推导）
  - softmax 减去 最大数字不变的证明
	

## 10、Dropout

  - Dropout 的原理。为什么能防止过拟合？代码实现
  - Dropout 在训练和测试的区别，怎么保证测试结果稳定
  - Dropout的随机因子会对结果的损失有影响吗
  - Dropout是失活神经元还是失30、什么是梯度爆炸？
  - 直接简化网络和dropout的区别
  - ResNet为什么不用Dropout
	   https://www.zhihu.com/question/325139089/answer/688743474

## 11、批归一化Batch Normalization原理

  - BN 可以防止过拟合么？为什么 
  - BN 层的原理，为什么要加缩放和偏置？ 相关公式，优化过程，优化的是什么，为什么BN有泛化能力的改善. 
  - BN有哪些需要学习的参数, BN前向、后向计算
  - BN 在训练和测试的区别？  
  - BN如何在inference是加速
  - 均值和方差，在测试和训练时是怎么获得的, BN在inference的时候用到的mean和var怎么来的：类似于滑动平均
  - BN跨卡训练怎么保证相同的mean和var， 问了面试官是SYNC
  - BN、LN、IN、GN原理及适用场景,共性和特性
  - BN和GN的区别？各有什么优缺点？
  - 如果数据不是高斯分布，bn后怎么恢复
  - bn的可训练参数 
  - BN和普通的Normalization的区别
  - BN放在激活函数前后有什么区别？
  - BN的gama  labada意义
  - 小batch size的坏处
  - BN前向、后向计算
  - ResNet 等模型中，Batch Normalization和Dropout 为什么不同时使用（同Dropout 中问题）
  - dropout和BN 在前向传播和方向传播阶段的区别？
	

## 12、- 深度学习训练

  - 深度学习中的batch的大小对学习效果有何影响？
  - batch size是不是越大越好？为什么
  - 深度学习batch size怎么训练，不考虑硬件约束
  - 如果加大batch size需要调节哪些超参数(答加大learning rate，增加epoch) 
  - 如果网络初始化为0的话有什么问题

## 13、-- 梯度消失和梯度爆炸：

  - ResNet 残差网络解决的是什么问题（网络退化，现象、原因和解决方法）？为什么能解决？
  - 梯度消失与梯度爆炸原因？解决方案
  - 从数学层面分析一下
  - 梯度下降现在为什么用的少
  - 解决梯度消失一般都用什么损失函数
  - ResNet如何解决梯度消失？公式写一下
  - resnetV1-V2 两种结构具体怎么实现
  
## 14、过拟合欠拟合

  - 如何判断过拟合，以及解决方法
  - 过拟合、欠拟合(overfitting,underfitting) 的原因及解决方法,解释为什么有效 
     	https://blog.csdn.net/weixin_43455338/article/details/104885402
  - BN 不能彻底解决过拟合的原因 --- https://www.zhihu.com/question/275788133

## 15、数据不平衡

  - 如何解决机器学习中的数据不平衡问题，训练数据类别样本不平衡的解决办法 -- https://www.zhihu.com/question/66408862/answer/243584032
  - 数据不均衡有什么解决方式，从数据，模型选择，以及损失函数选择角度
  - 如何解决深度学习中的数据不平衡问题(重点考核损失函数优化，正负样本不均衡时的解决方案--Focal-Loss 、GHM)  
  - 如果把不平衡的训练集(正负样本1：3) 通过降采样平衡后，那么对于平衡后的AUC值和预测概率值有怎样的变化；
  - 如何进行数据预处理，如何进行数据归一化
  - 为什么要归一化？(消除数据量纲差，可以剔除一些异常值，会使得模型收敛快一些也好一些，计算友好度也会稍微好一些) 
  - 数据增强
  - 如果训练集不平衡，测试集平衡，直接训练和过采样欠采样处理，哪个更好
  - 数据清洗方面有什么优化的想法)
  - 有哪些可以解决目标检测中正负样本不平衡问题的方法
  - 小样本问题如何解决

	
## 16、目标检测 + 小目标检测

  - 图片检测、识别、分割的区别
  > 进神经网络时第一个不同是，识别可以resize但是检测对位置敏感不能resize（传统上说，现在可以resize，但是记住缩放信息，能够还原）

  - 目标检测算法中多尺度训练/测试是怎么实现的? 
  - bbox目标大小尺寸差异巨大怎么解决？损失函数上怎么设计？
  - 介绍一下目标检测中的多尺度训练/测试
  - 目标检测中如何解决目标尺度大小不一的情况 (图像金字塔，特征金字塔，Inception block)
  - 如何提高小目标检测？  https://www.cnblogs.com/E-Dreamer-Blogs/p/11442927.html
  - 小目标检测有哪些trick

## 17、Faster-rcnn相关

  - RCNN系列模型的区别， Faster R-CNN网络做了哪些改进/优化
  - 项目中使用Faster rcnn，请问Faster rcnn的优势是什么，为什么在这个项目使用Faster rcnn
  - Faster-rcnn RPN的作用和原理，RPN怎么计算 box 的实际坐标
  - RPN 网络损失函数(多任务损失:二分类损失+SmoothL1损失)
  - RPN损失中的回归损失部分输入变量是怎么计算的？(注意回归的不是坐标和宽高，而是由它们计算得到的偏移量)
  - 原始图片中的RoI如何映射到到feature map？    https://zhuanlan.zhihu.com/p/24780433

  
  - 目标检测之Loss：Faster-RCNN中的Smooth L1 Loss ：https://blog.csdn.net/ytusdc/article/details/86301859

  - 为什么Faster-rcnn、SSD中使用Smooth L1 Loss 而不用Smooth L2 Loss ：https://blog.csdn.net/ytusdc/article/details/86659696

  - 说一下RoI Pooling是怎么做的？有什么缺陷？有什么作用
  > 优点： 1.允许我们对CNN中的feature map进行reuse；2.可以显著加速training和testing速度；3.允许end-to-end的形式训练目标检测系统。
  - Faster R-CNN是如何解决正负样本不平衡的问题？
  > 限制正负样本比例为1:1，如果正样本不足，就用负样本补充，这种方法后面研究工作用的不多。通常针对类别不平衡问题可以从调整样本数或修改loss weight两方面去解决，常用的方法有OHEM、OHNM、class balanced loss和Focal loss。
  - faster-rcnn中bbox回归用的是什么公式，说一下该网络是怎么回归bbox的？
  - RoI Pooling和RoI Align区别, 顺便介绍三种图像插值方法
  - anchor_bbox如何恢复到原始的大小，写一下推理过程 
  - Faster rcnn anchor机制，分别说一下 RPN阶段两种Loss分别是什么？
  - 如何从rpn网络生成的多个候选框中确定出目标候选框
  - Faster-rcnn有什么不足的地方吗？如何改进？faster-rcnn怎么优化
  - faster-rcnn 损失函数，优化函数，为什么回归损失中用smooth L1 (faster-rcnn) 
  - Faster R-CNN 训练和测试的流程有什么不一样
  - Fast-rcnn的区域候选框是怎么得到的
  - FPN结构，FPN对于多尺度的目标能比较好的原因    
  - SSD、yolo、Fast RCNN 的区别
  
  - 简要阐述一下One-Stage、Two-Stage模型
  - Faster RCNN和SSD有啥不同，为啥SSD快？(不做Region Proposal，one-stage的) 
  - Fast RCNN、yolo和ssd中正样本怎么确定的   https://blog.csdn.net/xiaotian127/article/details/104661466
  - YOLO的损失函数
  - YOLO的路由层作用是什么,  ①融合特征②开辟一个新的检测分支
  - YOLOV1~V4系列介绍，以及每一版的改进，优缺点介绍（越细越好）。
  - yolov3中的anchor怎么生成的, 写出 YOLOv3 的损失函数
  - YOLO中如何通过 K-Means 得到 anchor boxes？
  - YOLOv3中bbox坐标回归怎么做的？和Faster R-CNN有什么区别？  https://segmentfault.com/a/1190000021794637
  - YOLOv3中bbox坐标回归中的sigmoid函数有什么用？
  - YOLOv3中 route层的作用是什么？
  - yolov2中聚类是怎么做的
  - Anchor大小、长宽比选取？我说了业界常用的方法(YOLO9000中的方法) ，并提了一个更优的方法
  - 如果YOLOV3采用Focal loss会怎么样？
  - YOLOv3在小缺陷检测上也很好，RPN上和two-stage的有什么区别
  - yolo跟ssd的损失函数是什么样，有啥缺点，
  - YOLOv4用到哪些优化方法？https://blog.csdn.net/wonengguwozai/article/details/106784642
  - YOLOv4和YOLOv5有哪些区别？
  - YOLOv4相较于YOLOv3有哪些改进？速度更快还是更慢，为什么？
  
  - FPN的特征融合为什么是相加操作呢？

   >假设两路输入来说，如果是通道数相同且后面带卷积的话，add等价于concat之后对应通道共享同一个卷积核。FPN里的金字塔，是希望把分辨率最小但语义最强的特征图增加分辨率，从性质上是可以用add的。如果用concat，因为分辨率小的特征通道数更多，计算量是一笔不小的开销。所以FPN里特征融合使用相加操作可以理解为是为了降低计算量。
   - 基于FPN的RPN是怎么训练的？



  - 如何理解concat和add这两种常见的feature map特征融合方式
  > concat是通道数的增加;  add是特征图相加，通道数不变
你可以这么理解，add是描述图像的特征下的信息量增多了，但是描述图像的维度本身并没有增加，只是每一维下的信息量在增加，这显然是对最终的图像的分类是有益的。而concatenate是通道数的合并，也就是说描述图像本身的特征数（通道数）增加了，而每一特征下的信息是没有增加。
concat每个通道对应着对应的卷积核。 而add形式则将对应的特征图相加，再进行下一步卷积操作，相当于加了一个先验：对应通道的特征图语义类似，从而对应的特征图共享一个卷积核（对于两路输入来说，如果是通道数相同且后面带卷积的话，add等价于concat之后对应通道共享同一个卷积核）。
因此add可以认为是特殊的concat形式。但是add的计算量要比concat的计算量小得多。

  - 14.阐述一下目标检测任务中的多尺度
> 输入图片的尺寸对检测模型的性能影响相当明显，事实上，多尺度是提升精度最明显的技巧之一。在基础网络部分常常会生成比原图小数十倍的特征图，导致小物体的特征描述不容易被检测网络捕捉。通过输入更大、更多尺寸的图片进行训练，能够在一定程度上提高检测模型对物体大小的鲁棒性，仅在测试阶段引入多尺度，也可享受大尺寸和多尺寸带来的增益。
检测网络SSD中最后一层是由多个尺度的feature map一起组成的。FPN网络中采用多尺度feature map分层融合，分层预测的方法可以提升小目标的检测效果。
阐述一下如何进行多尺度训练
多尺度训练可以分为两个方面:一个是图像金字塔，一个是特征金字塔
1、人脸检测的MTCNN就是图像金字塔，使用多种分辨率的图像送到网络中识别，时间复杂度高，因为每幅图都要用多种scale去检测。2、FPN网络属于采用了特征金字塔的网络，一次特征提取产生多个feature map即一次图像输入完成，所以时间复杂度并不会增加多少3、faster rcnn多个anchor带来的多种尺寸的roi可以算muti scale思想的应用。

## 18、ResNet

  - Resnet提出的背景和核心理论
  - ResNet相关。描述、介绍特点、介绍为什么效果不会随着深度变差,   
  - ResNet解决了什么问题
  - ResNet为什么能解决梯度消失的问题
  > 网络退化问题： 训练深层的神经网络，会遇到梯度消失和梯度爆炸（vanishing/exploding gradients）的问题，影响了网络的收敛，但是这很大程度已经被标准初始化（normalized  initialization）和BN（Batch Normalization）所处理。 当深层网络能够开始收敛，会引起网络退化（degradation problem）问题，即随着网络深度增加，准确率会饱和，甚至下降。这种退化不是由过拟合引起的，因为在适当的深度模型中增加更多的层反而会导致更高的训练误差。 ResNet就通过引入深度残差连接来解决网络退化的问题，从而解决深度CNN模型难训练的问题。
  
  - ResNet V2 主要研究了什么问题， ResNet的BN层位置
  - resnet两种结构具体怎么实现，bottleneck的作用，为什么可以降低计算量，resnet参数量和模型大小
  - skip connection有什么好处？---- 推了下反向传播公式，根据链式法则，梯度可以直接作用于浅层网络。
  - 为什么 DenseNet 比 ResNet 更耗显存？
  - 相同层数，densenet和resnet哪个好，为什么？


## 19、Fcoal loss
  - RetinaNet 介绍， Focal loss， OHEM(online Hard example mining) 到底比focal loss差再哪里了
  - 如何解决前景背景数量不均衡
  - Focal Loss是如何进行难分样本挖掘的，有多少超参数？如何调参？
  - Focal Loss 与 GHM
  - IOU， GIOU，DIOU

 
## 20、轻量化模型
  - MobileNet v1 v2 介绍和区别，MobileNetV2中1x1卷积作用
  - MobileNet V2中的Residual结构最先是哪个网络提出来的，MobileNetV2 module的参数量和FLOPs计算
  - shufflenet 算法题：random_shuffle的实现
  - shuffle v1 v2 结构
  - 介绍一下组卷积
  - 深度可分离卷积 原理，为什么降低计算量，口述计算，减少了多少
  - 为什么mobileNet在理论上速度很快，工程上并没有特别大的提升？先说了卷积源码上的实现，两个超大矩阵相乘，可能是group操作，是一些零散的卷积操作，速度会慢。说应该从内存上去考虑。申请空间？
  - MobileNet系列为什么快？

   
   
## Anchor-free 目标检测-----
  - 介绍常见的 Anchor free 目标检测算法
  - 介绍Anchor based 和Anchor free目标检测网络的优缺点
  - CornerNet介绍，CornerPooling是怎么做的，怎么解决cornernet检测物体合并为一个框的问题  
  - CenterNet具体是如何工作的，介绍一下其损失函数
  - FCOS网络
 
## 网络结构 --
  - VGG，GoogleNet，ResNet等网络之间的区别是什么？
  - 介绍Inception(V1-V4) 网络结构以及优缺点

  - PAnet PSPNet   https://www.pianshen.com/article/6550689403/  
                    https://zhuanlan.zhihu.com/p/110204563

  - Unet, Unet 变体  https://blog.csdn.net/sinat_17456165/article/details/106132558
  - 卷积神经网络中的即插即用模块  https://blog.csdn.net/DD_PP_JJ/article/details/106436428

  
## 分割：
  
  - FCN网络为什么用全卷积层代替全连接层？ FCN与CNN最大的区别？
  - deeplab 系列， deeplab v3如何改进，训练过程, 介绍deeplabv3,画出backbone
  - 简述Deeplab v3网络相比于之前的v1和v2网络有哪些改进
   >①重新讨论了空洞卷积的使用，这让我们在级联模块和空间金字塔池化的框架下，能够获取更大的感受野从而获取多尺度信息。②改进了ASPP模块：由不同采样率的空洞卷积和BN层组成，我们尝试以级联或并行的方式布局模块。③讨论了一个重要问题：使用大采样率的3×3的空洞卷积，因为图像边界响应无法捕捉远距离信息，会退化为1×1的卷积, 我们建议将图像级特征融合到ASPP模块中。④阐述了训练细节并分享了训练经验。
  - 说一下deeplab。它与其他state of art的模型对比
  - deeplab的亮点是什么， 你认为deeplab还可以做哪些改进？
  - 介绍deeplabv3,画出backbone
  - 介绍金字塔池化，ASPP，深度可分，带孔卷积， PSPNet中PSP
  - 语义分割中CRF的作用,介绍一下 CRF的原理
  - HMM 和 CRF的区别
  - CRF 怎么训练的(传统+深度学习) 
  - 为什么深度学习中的图像分割要先编码再解码？
  - BN在图像分割里面一般用吗？
  - mask rcnn如何提高mask的分辨率，
  - deeplabv3的损失函数
  - 图像分割领域常见的损失函数
  - 剪枝压缩，些精简网络 (tplink) 
  - 介绍Mimic知识蒸馏是怎么做的
  - 语义分割评价指标 Miou
  - 串联与并联的ASPP都需画出。论文中认为这两种方式哪种更好？
   > 我答了并联更好，串联会产生Griding Efect。
      问：如何避免Griding Efect--网格效应（棋盘格效应）
  - 代码：mIOU(图像分割的通用评估指标) 的代码实现，使用numpy(我直接用了python) 
  - 分割小目标的经验
  - 全景分割中的stuff和things的区别
  - 语义分割的常见Loss及优缺点
  - 最新的分割网络框架了解吗- 如何划分训练集？如何选取验证集？
  - 为什么图像分割要先encode，再decode？
  - U-Net神经网络为什么会在医学图像分割表现好？
  

  - 9.分割出来的结果通常会有不连续的情况，怎么处理？开运算闭运算

  > 设定阈值，去掉阈值较小的连通集，和较小的空洞。
开运算 = 先腐蚀运算，再膨胀运算（看上去把细微连在一起的两块目标分开了）
开运算总结：（１）开运算能够除去孤立的小点，毛刺和小桥，而总的位置和形状不便。
（２）开运算是一个基于几何运算的滤波器。（３）结构元素大小的不同将导致滤波效果的不同。（４）不同的结构元素的选择导致了不同的分割，即提取出不同的特征。
闭运算 = 先膨胀运算，再腐蚀运算（看上去将两个细微连接的图块封闭在一起）
闭运算总结：（1）闭运算能够填平小湖（即小孔），弥合小裂缝，而总的位置和形状不变。
（2）闭运算是通过填充图像的凹角来滤波图像的。（3）结构元素大小的不同将导致滤波效果的不同。（4）不同结构元素的选择导致了不同的分割。



  - 介绍有监督、自监督和半监督
  
  
 
## 自注意力机制， Attention-----
  - 介绍自注意力机制
  - 介绍SENet中的注意力机制 --  Channel Attention  Squeeze-Excitation结构是怎么实现的？
  - 这里SEnet 采用sigmoid而不是softmax 为什么
     1、它要可以学习到各个channel之间的非线性关系 2、学习的关系不是互斥的，因为这里允许多channel特征，而不是one-hot形式。
  - Soft Attention 和 Hard Attention
  - 介绍CV方向上的注意力网络
  - Attention对比RNN和CNN，分别有哪点你觉得的优势
  - 写出Attention的公式
  - Attention机制，里面的q,k,v分别代表什么
  - 谈谈 Soft Attention，Attention 中需要先线性变换么？ 
  - 写一下Self-attention公式，Attention机制
  - 为什么self-attention可以替代seq2seq
  - Attention里面的QKV都是什么，怎么计算的

 - 介绍EfficientNet
 - RNN为什么不能解决长期依赖的问题？




 - 了解维度爆炸吗
 - 神经网络节点太多如何加快计算？
 - LSTM为什么能解决梯度消失/爆炸的问题

  余弦相似度距离和欧氏距离的区别， 你知道其他距离度量公式啊？
- 训练深度学习网络时候，出现Nan是什么原因，怎么才能避免？
- 如何解决训练集和测试集的分布差距过大问题？
- pytorch 多卡训练 同步还是异步
- 项目中用到图像的分辨率是多少
 -说下平时用到的深度学习的trick
15、怎么在我原有的结果上提升准确率

①提高数据质量，数据扩充，数据增强（mixup training）。②改变网络结构③改变优化器，改变学习率④知识蒸馏？



15.如果有很长，很小，或者很宽的目标，应该如何处理目标检测中如何解决目标尺度大小不一的情况 小目标不好检测，有试过其他的方法吗？比如裁剪图像进行重叠
小目标不好检测的两大原因：1）数据集中包含小目标的图片比较少，导致模型在训练的时候会偏向medium和large的目标。2）小目标的面积太小了，导致包含目标的anchor比较少，这也意味着小目标被检测出的概率变小。

改进方法： 1）对于数据集中含有小目标图片较少的情况，使用过度采样（oversample）的方式，即多次训练这类样本。2）对于第二类问题，则是对于那些包含小物体的图像，将小物体在图片中复制多分，在保证不影响其他物体的基础上，人工增加小物体在图片中出现的次数，提升被anchor包含的概率。3）使用FPN；4）RPN中anchor size的设置一定要合适，这样可提高proposal的准确率。5）对于分辨率很低的小目标，我们可以对其所在的proposal进行超分辨率，提升小目标的特征质量，更有利于小目标的检测。

场景分析--
  - 训练集loss上升，验证集loss保持基本不变，为什么
  - 关于神经网络的调参顺序?
  - 出现漏检、误检，怎么解决？
  - 如何训练模型、调优
  - 零样本分类问题。如果测试时出现一个图片是训练时没有的类别，怎么做
  - 介绍你知道的调参tricks  https://www.zhihu.com/question/41631631
  
  - 如何划分训练集？如何选取验证集？
  
  - 数据样本不均衡条件下：调参指标不能用accuraty ，因为本身就不平衡，应该用auc（常用）或者f1-score
  - auc物理含义 对正样本预估的概率 大于 对负样本预估的概率 的概率
  
  - 梯度爆炸会引发什么问题？
    
	在深度多层感知机网络中，梯度爆炸会引起网络不稳定，最好的结果是无法从训练数据中学习，而最坏的结果是出现无法再更新的 NaN 权重值。
    在循环神经网络中，梯度爆炸会导致网络不稳定，无法利用训练数据学习，最好的结果是网络无法学习长的输入序列数据。
  
  - 如何确定是否出现梯度爆炸？
    

code 编程
 - 卷积底层的实现方式(如caffe里面的img2col) 
 - 手撕 IoU,NMS, 及其变体 SoftNMS代码, softmax 解决了什么问题。soft nms的具体过程
 - 写一下mAP公式
 - 如何计算 mIoU？
 - 解释mAP，具体怎么计算？
 - nms很耗时吗？ 时间复杂度？ 一般预测时会有多少个候选框？
 - numpy实现交叉熵
 - 例如计算flops，卷积维度变换的公式推导，卷积是如何编程实现的；
 - pytorch中多卡训练的过程是怎样的？说下gather scatter是怎么做的？
 - 多卡训练的时候batchsize变大了精度反而掉了，这是为什么？有想过怎么解决吗？
 - 每张卡都有模型的话BN的参数一样吗？
 - 设计一个在CNN卷积核上做dropout的方式
 - PyTorch的高效convolution实现
 - PyTorch 不用库函数如何实现多机多卡
 - dataloader 简单写， 自己实现pytorch里面的dataloader，你怎么可以使它加载快点
 - 用 PyTorch写一下大致的train val的流程
 - 手写Resnet的跳转连接(pytorch)，以类的形式封装好，后续别的网络模块可直接调用
 
  TensorRT
  ONNX
  tensorflow和pytorch的区别
  pytorch generate 多线程

pytorch 多卡训练 同步还是异步
Pytorch多GPU数据流


余弦相似度距离和欧氏距离的区别？
