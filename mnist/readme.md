## 数据集下载
数据集下载地址：http://yann.lecun.com/exdb/mnist/

```
mkdir train_data
cd train_data
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
gzip -d ##解压两个文件
mkdir test_data
cd test_data
wget ##下载并解压两个训练集合
```

## Train
### FP32 train
```
export OMP_NUM_THREADS=28
batchsize = 128
numactl -c 0
BS：128
aveagre step time: 8.421220840551914 ms
average thoughput is: 15199.696388868371 fps

MKLDNN profiling:
primtive                time (ms)               calls
convolution_backward_weights    454.604117499999        936
convolution             434.2727300000002       940
convolution_backward_data    348.04083100000037      468
inner_product           260.01903920000035      940
pooling_backward_data    247.58322709999987      936
reorder                 137.15405687600307      5154
pooling                 109.55321329999988      940
eltwise_backward_data    98.30980248000009       1404
softmax                 2.1625971299999938      468

total time: 2091.699614586003 ms for 9 items.
```
1个epoch一共468步，每步，总共消耗了8.4ms， MKLDNN消耗了4.5 ms

### BF16 train
* 第一种方法，在所有权重和输入后面加上cast成BF16的操作，前向和方向计算的时候是BF16，最后反向梯度累加到权重上的时候，先cast回FP32再累加到权重上面
* 第二种方案，所有权重都直接定义成BF16的格式，这样梯度直接是BF16累加到权重上。因为累加的时候，浮点数相加有阶数对齐的一步，BF16格式小数位比较少，会丢失BF16的精度
所以我们采用第一种方案，在所有fp32变量(卷积权重，bias)的后面接一个cast成BF16的操作，而不是直接将变量定义成BF16

##### Enable的步骤
1. 利用变量域的cast 属性custom_getter，在变量后面接一个cast成BF16的操作，这样可以将所有的variable后面都加上了cast的操作
```
    def custom_dtype_getter(getter, name, shape=None, dtype=tf.float32, *args, **kwargs):
        var = getter(name, shape, tf.float32, *args, **kwargs)
        return tf.cast(var, dtype=tf.bfloat16, name=name + '_cast')
        #return getter(name + '_suffix', *args, **kwargs)
    with tf.compat.v1.variable_scope("alexnet_model", custom_getter=custom_dtype_getter):
```
为了使其生效，一定用tf.compat.v1.get_variable方法定义变量，不能使用tf.compat.v1.Variable 方法定义变量
这样所有的variable之后都会加上cast操作了

2. 输入的X需要显性加一个cast成BF16的操作（或者直接将输入这个placehodler设定为BF16, Y(label)不加，因为在softmax之前需要先将BF16转成FP32，做softmax和后面的loss的计算，因为目前softmax不支持BF16运算

```
BS：128
aveagre step time: 10.476168404277573 msec
average thoughput is: 12218.207560288523 fps

MKLDNN profiling:
primtive                time (ms)               calls
convolution_backward_data    680.5820520000002       936
matmul                  442.87427590000055      2812
convolution_backward_weights    434.78713660000005      936
convolution             332.1064502000002       940
reorder                 112.07055616400142      5622
pooling_backward_data    91.29346390000008       936
eltwise_backward_data    45.03222709999997       1404
eltwise                 23.26342000000018       1410
pooling                 22.1520989              940
softmax                 2.653808529999998       470

total time: 2186.815489294003 ms for 10 items.
```
1个epoch一共468步，每步，总共消耗了10.4ms， MKLDNN消耗了4.67 ms

目前BF16和FP32性能差不多，因为BF16加了cast操作，而且模型比较浅


## Inference
### FP32 inference
```
BS：28
average thoughput is: 20716.31 fps

MKLDNN profiling:
primtive                time (ms)               calls
convolution             111.73267700000021      714
pooling                 14.395259200000002      714
eltwise                 3.155272720000005       357
reorder                 3.0217281299999947      359
softmax                 1.2001958399999972      357

total time: 133.5051328900002 ms for 5 items.
```
1个epoch一共357步，每步，总共消耗了1.27ms， MKLDNN消耗了0.37 ms

### BF16 inference
有两种方式:
#### 方式1：
直接用BF16 training得到的模型，参考inference_bf16_method1.py
这种方式简单，但是缺点是还是保留了很多cast的操作

#### 方式2：
将FP32训练得到的图的权重直接全部转成BF16的格式
这种方式需要多一步转换的步骤，但是省去了图中 cast的操作 参考inference_bf16_method2.py 里面有个类，针对这个模型做了BF16的转换
遍历一次图，将所有权重改为BF16格式
* 所有输入placehold和变量freeze出来constant都变为BF16（int32的constant不转换 比如reshape和pad的输入）
* softmax的前面加一个BF16向FP32的cast
* 其余所有op的类型T都变为BF16

似乎第一种方法，在inference的时候cast会和const做一次 fold-constant的图优化，所以两者的性能相差不大
但是理论上看，肯定第二种方法的性能比较好

### INT8 inference
将FP32模型做一次 量化计算过程，计算scaling因子，并生成INT8 计算图

基本思想：
* 对于输入的张量
每一个FP32的输入张量，额外通过一个Min Op得到最小值Min，通过一个Max op得到最大值Max。原始FP32张量，和Min以及Max一起过一个quantize的op得到INT8的张量，再过INT8的计算op(POOL,Conv2D)。得到INT8的计算结果，将INT8的计算结果和output的Min以及output的Max值一起过一个Dequantize的op反量化得到FP32的输出
如果邻近两个节点都是INT8的量化操作，它们之间的反量化和量化操作可以省略
* 对于原来存储的FP32格式的weight以及bias
直接INT8化存储就可以了，存INT8值以及Min以及Max
* requantize 的操作，因为VNNI的计算结果是INT32，输出结果需要requantize 成INT8，这一步量化工具第二次遍历的时候会做掉
你会看到量化计算的节点的输入还有min_freezed_output，max_freezed_output 就是记录了INT32向INT8 re-quantize的scaling

所以量化节点的输入，包括FP32节点的所有输入，每个FP32输入的min和max scaling以及输出(INT32转INT8)requantize的min和max
量化节点的输出包括 INT8输出，转成FP32的min和max（这两个值会作为下一个节点的输入用在计算中）

在输入节点的后面，会接一个QuantizeV2节点专门量化
在softmax的前面有一个dequantize的节点反量化

**更细节的内容** 看自己tensorflow代码阅读的记录

运行INT8计算图进行inference

## TVM
```
mkdir export
python mnist_tvm.py #将fp32 pb 模型转成tvm module保存到export目录

python mnist_tvm_inference.py #运行tvm module 应该和fp32一样的精度
```