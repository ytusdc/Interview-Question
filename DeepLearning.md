# 深度学习相关
====

卷积输出计算，卷积核参数数量

1、图片检测、识别、分割的区别 ---- 进神经网络时第一个不同是，识别可以resize但是检测对位置敏感不能resize（传统上说，现在可以resize，但是记住缩放信息，能够还原）
	- 手推梯度反向传播
	- 介绍熟悉的NASNet网络
	- 常用的凸优化方法，介绍
## 2、CNN

	-- BP 算法手推反向传播，cnn怎么更新w值和反向传播， 手推CNN公式
	-- BP神经网络的结构是怎样的？和其他神经网络的区别是什么？
	-- CNN反向传播公式推导
    -- CNN的平移不变性是什么？如何实现的？CNN网络中的不变性理解
    -- CNN是深度深好还是长宽大好
    -- 神经网络怎样进行参数初始化？
	-- CNN 模型所需的计算力（flops）和参数（parameters）数量是怎么计算的？
	
	
### 3、-- 卷积相关

   	-- 1 * 1卷积核的作用，哪些情况可以使用1x1卷积？
   	-- 3×3 卷积核 与 5×5 卷积核相比的优点，卷积操作是线性的吗？CNN是线性的吗？为什么？
	-- 感受野的计算，CNN 的感受野受什么影响
   	-- 卷积输出计算，卷积核参数数量计算：普通卷积、DW PW卷积计算量推导，计算flops，网络参数量计算
   	-- 实现卷积操作(代码) 
   	-- 卷积神经网络的卷积核大小、个数,卷积层数如何确定呢?
	-- deformable conv怎么做,具体怎么学的，对偏移有没有什么限制
	-- 卷积神经网络的卷积核为什么是方形？为什么是奇数
   	-- 空洞卷积及其优点
    -- 反卷积具体怎么实现的？
	-- 卷积如何加速
	-- 卷积层如何修剪，量化等
	上采样、下采样还有哪些类型
	
	卷积是降采样过程
	
    upsample 上采样方法(语义分割)
	
	1、up-sampling--（Billnear intrpolation双线性插值）， 临近插值
    2、Transpose Conv （反卷积）
    3、Up-Pooling
	-介绍常见的插值算法

	
#### 4、--Pool 池化的作用和种类

	(1) Max Pooling和 Average Pooling的区别，使用场景分别是什么？
	(2) 哪些情况用 MaxPool比AveragePool效果好？原因
	(3) pooling怎么反向传播
	(4) RoI Pooling和RoI Align区别,  RoI Pooling 哪些操作导致其精度没有RoI Align高	
	(5) 当出现一个大噪点时，用哪种池化更好
	(6) 除了池化层还有什么方法可以减少特征数量(这里他提到了UNet模型，之前有看过，就说出来了，是采样的方法) 
	(7) anchor_bbox如何恢复到原始的大小，写一下推理过程。
	(8) RoI Pooling的计算过程，写一下推理过程。

5、--模型评估方法

	-- 精确率，召回率，和准确度评价怎么算，这俩是矛盾的怎么选最优
    -- ROC曲线和AUC曲线意义，ROC曲线上每个点代表的含义， 介绍F1-score, auc比F1好在哪,
	    AUC原理，为什么更适用于排序问题？  AUC怎么算的？ROC曲线怎么画的
	-- auc比F1好在哪-- ROC曲线 vs Precision-Recall曲线，各自的使用场景选择
	-- Log Loss 和 AUC 的区别， 适用于什么场景
	-- 分类、检测、分割评价指标说一下？
	-- 手写AUC曲面面积的计算(或者伪代码)
	-- 混淆矩阵
	-- AP和mAP的区别？
	-- AUC指标有什么特点？放缩结果对AUC是否有影响？
	-- 余弦距离与欧式距离有什么特点？https://www.zhihu.com/question/19640394
	-- 什么是偏差和方差
	
	
6、--各种激活函数的优缺点

	(1) 激活函数有什么用？各种激活函数（sigmoid，tanh，ReLU, leaky ReLU, PReLU, ELU （ReLU变体））介绍一下，优缺点及适用场景
	    --Relu不可导怎么办
	(3) 深度学习用relu激励函数，为什么，好处是什么
	(2) 比如sigmod的问题在哪里，relu是怎么解决的，relu的问题在哪里，有没有对应的解决算法(prelu) ，介绍 Leaky Relu 并写公式 leakyrelu解决了梯度消失问题吗
	(4) relu激励函数训练数据时会让神经元失活，训练之后进行应用时有什么注意的吗，训练时失活的神经元在应用时怎么办
	(5) 推导sigmoid的求导
	--  sigmoid优点，为什么用在最后一层

7、-- 损失函数(Loss) 

	(1) 常用的 Loss 函数 (MSE /huber loss/BCE/cross entropy/指数损失/smooth l1 )  
	(2) cross entropy的原理是什么？反向传播机制是什么？交叉熵的公式伪代码
	(3) 为什么分类用交叉熵而不用MSE(均方误差)，同为什么使用交叉熵作为损失函数去评估误差
    (4) 为什么损失函数用交叉熵
	(5) 多标签分类怎么解决，从损失函数角度考虑	
	(6) Loss不降、不收敛的原因和解决方法
	(7) 损失函数正则项的本质是什么?
	(8) 样本不均衡怎么搞(重点考核损失函数优化 focal loss) 
	(9) 分割网络的损失函数
	
	
	
8、- 深度学习常用的optimizer(优化器) 

	(1) 介绍一下你经常用的optimize
	(2) 凸优化了解吗？牛顿法、SGD、最小二乘法，各自的优势。---  SGD与牛顿法的区别在哪？
	(3) loss优化的几个方法(sgd、动量、adam) 
	(4) Adam优化器的迭代公式, AdaGrad梯度的原理
	(5) 动量法的表达式，随机梯度下降相比全局梯度下降好处是什么
	(6) SGD每步做什么，为什么能online learning
	(7) 你一般用哪个优化器，为什么用它？介绍你知道的优化器
    (9) 学习率调节策略	
    (10)为什么动量可以走出局部最小值？数学解释
    训练时出现loss NAN的可能因素	
	-- adam用到二阶矩的原理是什么
	梯度下降与拟牛顿法的异同？
	
9、--Softmax 相关

    (1) Softmax的原理是？反向传播、梯度公式推导，代码实现， 手推softmax的BP公式
	(2) 为什么softmax是指数形式
	(3) LR + softmax做多分类区别, 为什么 LR 用交叉熵损失而不是平方损失？
	(4) softmax、多个logistic的各自的优势？
	(5) 介绍分层 softmax，还有负采样？怎么负采样？
    (6) softmax 怎么防止溢出， softmax减去最大数字不变的证明
    (7) softmax得出的结果是排序的么，为什么分类用交叉熵
	(8) One-hot有什么作用？
	(9) 交叉熵和最大似然估计的关系推导）
	
	

10、--正则化方法有哪些

	(1) L0、L1、L2定义，L1，L2正则化原理，从参数分布和参数图上解释
	(2) L1，L2 norm的区别，L1为什么能使特征稀疏，L2为什么不能使特征稀疏，L2为什么能解决(或者减轻) 过拟合
	   (L1范数，使权重为0，对应的特征则不起作用，使特征稀疏稀疏矩阵) 
	(3) Lasso、线性回归、逻辑回归、l1 l2 正则有什么影响，
	(4) 理解：L1正则先验分布是Laplace分布，L2正则先验分布是Gaussian分布
	(5) 口述一下l1参数分布的推导(牛皮) , l1在0处不可导，怎么处理:利用坐标轴下降法或者proximal operator	：http://roachsinai.github.io/2016/08/03/1Proximal_Method/
	-- L1是损失函数，有哪些优化方法，能用sgd么？为什么？
	-- L1是不可导的，真的可以用么？
	-- L1有什么缺点？

11、- Dropout

	-- Dropout 的原理。为什么能防止过拟合？代码实现
	-- Dropout 在训练和测试的区别，怎么保证测试结果稳定
	-- Dropout的随机因子会对结果的损失有影响吗
	-- Dropout是失活神经元还是失30、什么是梯度爆炸？
	-- 直接简化网络和dropout的区别
	-- ResNet为什么不用Dropout
	   https://www.zhihu.com/question/325139089/answer/688743474

12、批归一化Batch Normalization原理

    -- BN 可以防止过拟合么？为什么 
	BN 层的原理，为什么要加缩放和偏置？ 相关公式，优化过程，优化的是什么，为什么BN有泛化能力的改善. 
	-- BN有哪些需要学习的参数, BN前向、后向计算
	-- BN 在训练和测试的区别？  
	-- BN如何在inference是加速
	-- 均值和方差，在测试和训练时是怎么获得的, BN在inference的时候用到的mean和var怎么来的：类似于滑动平均
        BN跨卡训练怎么保证相同的mean和var， 问了面试官是SYNC
	-- BN、LN、IN、GN原理及适用场景,共性和特性
	   -- BN和GN的区别？各有什么优缺点？
	-- 如果数据不是高斯分布，bn后怎么恢复
	-- bn的可训练参数 
	-- BN和普通的Normalization的区别
	-- BN放在激活函数前后有什么区别？
	-- BN的gama  labada意义
	-- 小batch size的坏处
	     BN前向、后向计算
	-- ResNet 等模型中，Batch Normalization和Dropout 为什么不同时使用（同Dropout 中问题）
	dropout和BN 在前向传播和方向传播阶段的区别？
	
	


13、- 深度学习训练

	(1) 深度学习中的batch的大小对学习效果有何影响？
	(2) batch size是不是越大越好？为什么
	(3) 深度学习batch size怎么训练，不考虑硬件约束
	(4) 如果加大batch size需要调节哪些超参数(答加大learning rate，增加epoch) 
	(5) 如果网络初始化为0的话有什么问题

14、-- 梯度消失和梯度爆炸：

	(1) ResNet 残差网络解决的是什么问题（网络退化，现象、原因和解决方法）？为什么能解决？
	(2) 梯度消失与梯度爆炸原因？解决方案
	(3) 从数学层面分析一下
	(4) 梯度下降现在为什么用的少
	(5) 解决梯度消失一般都用什么损失函数
	(6) ResNet如何解决梯度消失？公式写一下
	(7) resnetV1-V2 两种结构具体怎么实现
  
15、

    - 如何判断过拟合，以及解决方法
    - 过拟合、欠拟合(overfitting,underfitting) 的原因及解决方法,解释为什么有效 
     	https://blog.csdn.net/weixin_43455338/article/details/104885402
	- BN 不能彻底解决过拟合的原因 --- https://www.zhihu.com/question/275788133

16、--数据不平衡

	-- 如何解决机器学习中的数据不平衡问题，训练数据类别样本不平衡的解决办法 -- https://www.zhihu.com/question/66408862/answer/243584032
	-- 数据不均衡有什么解决方式，从数据，模型选择，以及损失函数选择角度
	-- 如何解决深度学习中的数据不平衡问题(重点考核损失函数优化，正负样本不均衡时的解决方案--Focal-Loss 、GHM)  
	-- 如果把不平衡的训练集(正负样本1：3) 通过降采样平衡后，那么对于平衡后的AUC值和预测概率值有怎样的变化；
    -- 如何进行数据预处理，如何进行数据归一化
	-- 为什么要归一化？(消除数据量纲差，可以剔除一些异常值，会使得模型收敛快一些也好一些，计算友好度也会稍微好一些) 
	-- 数据增强
	-- 如果训练集不平衡，测试集平衡，直接训练和过采样欠采样处理，哪个更好
	-- 数据清洗方面有什么优化的想法)
	-- 有哪些可以解决目标检测中正负样本不平衡问题的方法
	-- 小样本问题如何解决

	
## 目标检测 + 小目标检测

	- 目标检测算法中多尺度训练/测试是怎么实现的? 
	-- bbox目标大小尺寸差异巨大怎么解决？损失函数上怎么设计？
	-- 介绍一下目标检测中的多尺度训练/测试
    -- 目标检测中如何解决目标尺度大小不一的情况 (图像金字塔，特征金字塔，Inception block)
	-- 如何提高小目标检测？  https://www.cnblogs.com/E-Dreamer-Blogs/p/11442927.html
	-- 小目标检测有哪些trick


    
18、Faster-rcnn相关

    -- RCNN系列模型的区别， Faster R-CNN网络做了哪些改进/优化
	-- 项目中使用Faster rcnn，请问Faster rcnn的优势是什么，为什么在这个项目使用Faster rcnn
	-- Faster-rcnn RPN的作用和原理，RPN怎么计算 box 的实际坐标
	-- 原始图片中的RoI如何映射到到feature map？    https://zhuanlan.zhihu.com/p/24780433
    -- ROI pooling 的主要作用是什么（图片不同尺寸输入）？
	 RoI Pooling和RoI Align区别, 顺便介绍三种图像插值方法
    -- Faster rcnn anchor机制，分别说一下 RPN阶段两种Loss分别是什么？
	-- 如何从rpn网络生成的多个候选框中确定出目标候选框
    -- Faster-rcnn有什么不足的地方吗？如何改进？faster-rcnn怎么优化
	-- faster-rcnn 损失函数，优化函数，为什么回归损失中用smooth L1 (faster-rcnn) 
	-- Faster R-CNN 训练和测试的流程有什么不一样
	-- Fast-rcnn的区域候选框是怎么得到的
	-- FPN结构，FPN对于多尺度的目标能比较好的原因    
	-- SSD、yolo、Fast RCNN 的区别
	-- Faster RCNN和SSD有啥不同，为啥SSD快？(不做Region Proposal，one-stage的) 
	-- Fast RCNN、yolo和ssd中正样本怎么确定的   https://blog.csdn.net/xiaotian127/article/details/104661466
	-- YOLO的损失函数
	-- YOLO的路由层作用是什么,  ①融合特征②开辟一个新的检测分支
	-- YOLOV1~V4系列介绍，以及每一版的改进，优缺点介绍（越细越好）。
	-- yolov3中的anchor怎么生成的, 写出 YOLOv3 的损失函数
	-- YOLO中如何通过 K-Means 得到 anchor boxes？
	-- YOLOv3中bbox坐标回归怎么做的？和Faster R-CNN有什么区别？  https://segmentfault.com/a/1190000021794637
	-- YOLOv3中bbox坐标回归中的sigmoid函数有什么用？
	-- YOLOv3中 route层的作用是什么？
	-- yolov2中聚类是怎么做的
	-- Anchor大小、长宽比选取？我说了业界常用的方法(YOLO9000中的方法) ，并提了一个更优的方法
	-- 如果YOLOV3采用Focal loss会怎么样？
	-- YOLOv3在小缺陷检测上也很好，RPN上和two-stage的有什么区别
	-- yolo跟ssd的损失函数是什么样，有啥缺点，
	-- YOLOv4用到哪些优化方法？https://blog.csdn.net/wonengguwozai/article/details/106784642
	-- YOLOv4和YOLOv5有哪些区别？
	YOLOv4相较于YOLOv3有哪些改进？速度更快还是更慢，为什么？


# ResNet

- ResNet相关。描述、介绍特点、介绍为什么效果不会随着深度变差,   
  -- ResNet解决了什么问题
  -- ResNet为什么能解决梯度消失的问题
  -- 网络退化问题： 训练深层的神经网络，会遇到梯度消失和梯度爆炸（vanishing/exploding gradients）的问题，影响了网络的收敛，但是这很大程度已经被标准初始化（normalized  initialization）和BN（Batch Normalization）所处理。 当深层网络能够开始收敛，会引起网络退化（degradation problem）问题，即随着网络深度增加，准确率会饱和，甚至下降。这种退化不是由过拟合引起的，因为在适当的深度模型中增加更多的层反而会导致更高的训练误差。 ResNet就通过引入深度残差连接来解决网络退化的问题，从而解决深度CNN模型难训练的问题。
  
  -- ResNet V2 主要研究了什么问题， ResNet的BN层位置
  -- resnet两种结构具体怎么实现，bottleneck的作用，为什么可以降低计算量，resnet参数量和模型大小
  -- skip connection有什么好处？---- 推了下反向传播公式，根据链式法则，梯度可以直接作用于浅层网络。
  -- 为什么 DenseNet 比 ResNet 更耗显存？
  -- 相同层数，densenet和resnet哪个好，为什么？
  
 

Fcoal loss
  -- Focal loss --
  -- RetinaNet 介绍， Focal loss， OHEM(online Hard example mining) 到底比focal loss差再哪里了
	 如何解决前景背景数量不均衡
     Focal Loss是如何进行难分样本挖掘的，有多少超参数？如何调参？
	 Focal Loss 与 GHM
	 IOU， GIOU，DIOU

 
轻量化模型
  -- MobileNet v1 v2 介绍和区别，MobileNetV2中1x1卷积作用
  -- MobileNet V2中的Residual结构最先是哪个网络提出来的，MobileNetV2 module的参数量和FLOPs计算
  -- shufflenet 算法题：random_shuffle的实现
  -- shuffle v1 v2 结构
  介绍一下组卷积
  -- 深度可分离卷积 原理，为什么降低计算量，口述计算，减少了多少
  -- 为什么mobileNet在理论上速度很快，工程上并没有特别大的提升？先说了卷积源码上的实现，两个超大矩阵相乘，可能是group操作，是一些零散的卷积操作，速度会慢。说应该从内存上去考虑。申请空间？
  --  MobileNet系列为什么快？

   
   
Anchor-free 目标检测-----
  -- 介绍常见的 Anchor free 目标检测算法
  -- 介绍Anchor based 和Anchor free目标检测网络的优缺点
  -- CornerNet介绍，CornerPooling是怎么做的，怎么解决cornernet检测物体合并为一个框的问题  
  -- CenterNet具体是如何工作的，介绍一下其损失函数
  -- FCOS网络
 
网络结构 --
  -- VGG，GoogleNet，ResNet等网络之间的区别是什么？
  -- 介绍Inception(V1-V4) 网络结构以及优缺点

  -- PAnet PSPNet   https://www.pianshen.com/article/6550689403/  
                    https://zhuanlan.zhihu.com/p/110204563

  -- Unet, Unet 变体  https://blog.csdn.net/sinat_17456165/article/details/106132558
  -- 卷积神经网络中的即插即用模块  https://blog.csdn.net/DD_PP_JJ/article/details/106436428

  
分割：
  -- FCN网络为什么用全卷积层代替全连接层？
  -- deeplab v3如何改进，训练过程
  -- 说一下deeplab。它与其他state of art的模型对比
  -- deeplab的亮点是什么， 你认为deeplab还可以做哪些改进？
  -- 介绍deeplabv3,画出backbone
  -- 介绍金字塔池化，ASPP，深度可分，带孔卷积， PSPNet中PSP
  -- 语义分割中CRF的作用,介绍一下 CRF的原理
  -- HMM 和 CRF的区别
  -- CRF 怎么训练的(传统+深度学习) 
  -- 为什么深度学习中的图像分割要先编码再解码？
  -- BN在图像分割里面一般用吗？
  -- mask rcnn如何提高mask的分辨率，
  -- deeplabv3的损失函数
  -- 图像分割领域常见的损失函数
  -- 剪枝压缩，些精简网络 (tplink) 
  -- 介绍Mimic知识蒸馏是怎么做的
  -- 语义分割评价指标 Miou
  -- 串联与并联的ASPP都需画出。论文中认为这两种方式哪种更好？
      我答了并联更好，串联会产生Griding Efect。
      问：如何避免Griding Efect--网格效应（棋盘格效应）
  -- 代码：mIOU(图像分割的通用评估指标) 的代码实现，使用numpy(我直接用了python) 
  -- 分割小目标的经验
  -- 全景分割中的stuff和things的区别
  -- 语义分割的常见Loss及优缺点
  -- 最新的分割网络框架了解吗
  -- 为什么图像分割要先encode，再decode？
  U-Net神经网络为什么会在医学图像分割表现好？
  
 

  -- 介绍有监督、自监督和半监督
  
  
 
自注意力机制， Attention-----
  -- 介绍自注意力机制
  -- 介绍SENet中的注意力机制 --  Channel Attention  Squeeze-Excitation结构是怎么实现的？
  -- 这里SEnet 采用sigmoid而不是softmax 为什么
     1、它要可以学习到各个channel之间的非线性关系 2、学习的关系不是互斥的，因为这里允许多channel特征，而不是one-hot形式。
	 
  -- 介绍CV方向上的注意力网络
  -- Attention对比RNN和CNN，分别有哪点你觉得的优势
  -- 写出Attention的公式
  -- Attention机制，里面的q,k,v分别代表什么
  -- 谈谈 Soft Attention，Attention 中需要先线性变换么？ 
  
  -- 写一下Self-attention公式，Attention机制
  -- 为什么self-attention可以替代seq2seq
  -- Attention里面的QKV都是什么，怎么计算的

介绍EfficientNet
RNN为什么不能解决长期依赖的问题？




了解维度爆炸吗
神经网络节点太多如何加快计算？
  -- LSTM为什么能解决梯度消失/爆炸的问题


训练深度学习网络时候，出现Nan是什么原因，怎么才能避免？
- 如何解决训练集和测试集的分布差距过大问题？
pytorch 多卡训练 同步还是异步
项目中用到图像的分辨率是多少
--说下平时用到的深度学习的trick
15、怎么在我原有的结果上提升准确率

①提高数据质量，数据扩充，数据增强（mixup training）。②改变网络结构③改变优化器，改变学习率④知识蒸馏？


场景分析--
  -- 训练集loss上升，验证集loss保持基本不变，为什么
  -- 关于神经网络的调参顺序?
  -- 出现漏检、误检，怎么解决？
  -- 如何训练模型、调优
  -- 零样本分类问题。如果测试时出现一个图片是训练时没有的类别，怎么做
  --  介绍你知道的调参tricks  https://www.zhihu.com/question/41631631
  



code 编程
 -- 卷积底层的实现方式(如caffe里面的img2col) 
 -- 手撕 IoU,NMS, 及其变体 SoftNMS代码, softmax 解决了什么问题。soft nms的具体过程
 -- 写一下mAP公式
 -- 如何计算 mIoU？
 -- 解释mAP，具体怎么计算？
 -- nms很耗时吗？ 时间复杂度？ 一般预测时会有多少个候选框？
 -- numpy实现交叉熵
 -- 例如计算flops，卷积维度变换的公式推导，卷积是如何编程实现的；
 -- pytorch中多卡训练的过程是怎样的？说下gather scatter是怎么做的？
 -- 多卡训练的时候batchsize变大了精度反而掉了，这是为什么？有想过怎么解决吗？
 -- 每张卡都有模型的话BN的参数一样吗？
 -- 设计一个在CNN卷积核上做dropout的方式
 -- PyTorch的高效convolution实现
 -- PyTorch 不用库函数如何实现多机多卡
 -- dataloader 简单写， 自己实现pytorch里面的dataloader，你怎么可以使它加载快点
 -- 用 PyTorch写一下大致的train val的流程
 -- 手写Resnet的跳转连接(pytorch)，以类的形式封装好，后续别的网络模块可直接调用
 
  TensorRT
  ONNX
  tensorflow和pytorch的区别
  pytorch generate 多线程

pytorch 多卡训练 同步还是异步
Pytorch多GPU数据流


余弦相似度距离和欧氏距离的区别？
