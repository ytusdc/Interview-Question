## Python  直接赋值, 深拷贝，浅拷贝的区别， Python2 3的区别  
   
	直接赋值：其实就是对象的引用（别名）。
	浅拷贝(copy)：拷贝父对象，不会拷贝对象的内部的子对象。
	深拷贝(deepcopy)： copy 模块的 deepcopy 方法，完全拷贝了父对象及其子对象。

	详细连接：https://www.runoob.com/w3cnote/python-understanding-dict-copy-shallow-or-deep.html

## python的反射机制

	https://cloud.tencent.com/developer/article/1173302
	
## Python is 和 ==的区别  


	is 用于判断两个变量引用对象是否为同一个，就是所引用的对象的内存地址是否一致
	== 用于判断引用变量的值是否相等。只判断值和数据类型
	
	大家自己试试看a=257,b=257时它们的id还是否会相等。事实上Python 为了优化速度，使用了小整数对象池，避免为整数频繁申请和销毁内存空间。而Python 对小整数的定义是:(-5,257)只有数字在-5到256之间它们的id才会相等，超过了这个范围就不行了，同样的道理，字符串对象也有一个类似的缓冲池，超过区间范围内自然不会相等了。 

	https://zhuanlan.zhihu.com/p/59702679

 	https://www.cnblogs.com/kunpengv5/p/7811566.html
	
## Python的装饰器@的用法,装饰器用途和作用

## Python中的self关键字

	https://www.cnblogs.com/junge-mike/p/12761764.html

## Python的内存管理机制，优缺点，Python如何实现垃圾回收（引用计数）

	https://www.jianshu.com/p/518626051635

## 介绍一下python中闭包的作用

	https://www.cnblogs.com/s-1314-521/p/9763376.html
	
## python 函数传参会改变原值吗
  
	不可变对象：整型、字符串、元组（tuple)
  	可变对象： 列表（list）、集合（set）、字典(dictionary	

	当我们传的参数是不可变对象时，无论函数中对其做什么操作，都不会改变函数外这个参数的值；

	当传的是可变对象时，如果是重新对其进行赋值，则不会改变函数外参数的值，如果是对其进行操作，则会改变。即变量中存储的是引用 , 是指向真正内容的内存地址 , 对变量重新赋值 , 相当于修改了变量副本存储的内存地址 , 而这时的变量已经和函数体外的变量不是同一个了, 在函数体之外的变量 , 依旧存储的是原本的内存地址 , 其值自然没有发生改变 。  
	https://www.cnblogs.com/monkey-moon/p/9347505.html
	
## Python的多线程能否用来做并行计算？

	不能，它有GIL锁，但可以用多进程实现并行
	https://blog.csdn.net/liuweiyuxiang/article/details/99947468

## Python伪多线程，那什么时候应该用它？

	程序的性能受到计算密集型(CPU)的程序限制和I/O密集型的程序限制影响,那什么是计算密集型和I/O密集型程序呢?
	计算密集型(CPU)：--使用多进程
		高度使用CPU的程序， CPU计算占主要的任务 ,例如: 进行数学计算,矩阵运算,搜索,图像处理等.
	I/O密集型：--使用多线程
		I/0(Input/Output)程序是进行数据传输,例如: 文件操作,数据库,网络数据， 磁盘IO、网络IO占主要的任务，计算量很小， 请求网页、读写文件等
	
	GIL对I/O绑定多线程程序的性能影响不大,因为线程在等待I/O时共享锁.
	GIL对计算型绑定多线程程序有影响,例如: 使用线程处理部分图像的程序,不仅会因锁定而成为单线程,而且还会看到执行时间的增加,这种增加是由锁的获取和释放开销的结果.
	
	https://zhuanlan.zhihu.com/p/24283040
	https://blog.csdn.net/liuweiyuxiang/article/details/99947468


## tensorflow while_loop和python for循环的区别，什么情况下for更优？
	while loop的循环次数不确定的情况下效率低，因为要不断重新建图




  - readline和readlines的区别
  - 讲一下yield关键字？它的作用是啥？
  - xrange与range的区别
  - Python里面的lambda表达式写一下，随便写一个  fun = lambda x, y: x+y, fun(x, y)
  - Python里面的三元运算符写一下(x if x> y else y)
  - Python中的对象(Object)和C++中的对象有什么区别
    https://blog.csdn.net/fox64194167/article/details/79887116

1.用numpy 实现  m*2 的矩阵中 每一组数（xi,yi） 与 n*2 的矩阵的每一组 (xj，yj) 的欧氏距离之和， 即有 m*n 组欧氏距离， 其中 i 从 0 到 m-1 , j 从 0 到 n-1.


用numpy，pytorch，TensorFlow实现两个矩阵操作，不能用for循环（一个是给出两个矩阵，分别表示m个和n个向量，求第一个矩阵和第二个矩阵每个向量的夹角；二是求欧式距离）



