
### Pytorch损失函数总结

&emsp;&emsp;pytorch的损失函数有19个，但是目前常用的只有几个，因此在这里详细了解一下损失函数以及原理。

&emsp;&emsp;需要注意的是，损失函数都有三个参数

- size_average:Deprecated
- reduce:Deprecated
- reduction: 'none' | 'mean' | 'sum' 

&emsp;&emsp;在以下的测试代码中，参数都是默认的，默认参数都是求损失的mean。


#### 1. nn.L1Loss

$$loss(x_i,y_i) = |x_i - y_i|$$


```python
import torch
from torch import nn
loss_fn = nn.L1Loss()
test_input = torch.randn(1,2)
target = torch.randn(1,2)
print('input:' ,test_input)
print('target:',target)
output = loss_fn(test_input,target)
print('output:',output)
```

    input: tensor([[-0.7093,  0.6200]])
    target: tensor([[0.2530, 1.0503]])
    output: tensor(0.6963)


#### 2. nn.MSELoss

$$loss(x_i,y_i) = (x_i - y_i)^2$$


```python
loss_fn = nn.MSELoss()
test_input = torch.randn(1,2)
target = torch.randn(1,2)
print('input:' ,test_input)
print('target:',target)
output = loss_fn(test_input,target)
print('output:',output)
```

    input: tensor([[-0.1687,  0.1631]])
    target: tensor([[ 0.6217, -0.4989]])
    output: tensor(0.5315)


#### 3. nn.BCELoss

&emsp;&emsp;二分类用的交叉熵，用的时候需要在该层前面加上Sigmoid函数，其中$x_i$表示第i个样本预测为正例的概率。

$$loss(x_i - y_i) = -w_i [y_i log x_i + (1 - y_i)log(1 - x_i)]$$


```python
loss_fn = nn.BCELoss()
test_input = torch.randn(3)
target = target = torch.empty(3).random_(2)
print('input:' ,test_input)
print('target:',target)
output = loss_fn(torch.sigmoid(test_input),target)
print('output:',output)
```

    input: tensor([ 2.7002,  1.6961, -0.2694])
    target: tensor([1., 0., 0.])
    output: tensor(0.8323)


#### 4. nn.BCEWithLogitsLoss

&emsp;&emsp;上面的nn.BCELoss需要手动添加一个Sigmoid层，这里结合了两者，这样做能够利用 log_sum_exp trick，使得数值结果更加稳定

$$loss(x_i - y_i) = -w_i [y_i log x_i + (1 - y_i)log(1 - x_i)]$$


```python
loss_fn = nn.BCEWithLogitsLoss()
test_input = torch.Tensor([ 2.7002,  1.6961, -0.2694])
target = torch.Tensor([1., 0., 0.])
print('input:' ,test_input)
print('target:',target)
output = loss_fn(test_input,target)
print('output:',output)
```

    input: tensor([ 2.7002,  1.6961, -0.2694])
    target: tensor([1., 0., 0.])
    output: tensor(0.8323)


#### 5. nn.CrossEntropyLoss

&emsp;&emsp;多分类用的交叉熵损失函数，用这个loss前面不需要加Softmax层。

&emsp;&emsp;这里限制了target类型为torch.LongTenosor

- input: (N,C) where c = number of classes
- target: (N) where each value is 0 <= targets[i] <= c-1

$$loss(x,label) = -w_{label} log \frac{e^{x_{label}}}{\sum_{j=1}^{N} e^{x_j}} \\ 
= w_{label}[-x_{label} + log \sum_{j=1}^{N} e^{x_j}]$$

$$$$


```python
loss_fn = nn.CrossEntropyLoss()
test_input = torch.randn(3,5)
target = torch.empty(3, dtype=torch.long).random_(5)
print('input:' ,test_input)
print('target:',target)
output = loss_fn(test_input,target)
print('output:',output)
```

    input: tensor([[-1.2252, -0.4108,  1.2252, -0.4867,  0.6131],
            [ 0.3764, -2.4024,  1.8075, -1.4138, -0.8422],
            [ 1.4364,  0.1023,  0.2806, -0.8174, -0.6862]])
    target: tensor([4, 0, 2])
    output: tensor(1.5981)


#### 6. nn.NLLLoss

&emsp;&emsp;用于多分类的负对数似然损失函数

$$loss(x,label) = -x_{label}$$

&emsp;&emsp;在前面加上一个torch.LogSoftmax 层就等价于交叉熵损失了


```python
m = nn.LogSoftmax(dim=1)
loss_fn = nn.NLLLoss()
test_input = torch.randn(3,5)
target = torch.empty(3, dtype=torch.long).random_(5)
print('input:' ,test_input)
print('target:',target)
output = loss_fn(m(test_input),target)
print('output:',output)
```

    input: tensor([[-0.7851,  0.5215, -0.3608,  1.0634, -0.3758],
            [ 0.9265,  1.1597, -1.9180, -2.9440, -0.0021],
            [-0.7288,  0.5987,  1.9119,  0.5652, -0.1381]])
    target: tensor([1, 1, 4])
    output: tensor(1.5697)


#### 7. nn.KLDivLoss

&emsp;&emsp;KL散度，又叫做相对熵，计算的是两个分布之间的距离，越相似则越接近零。

$$loss(x,y) = \frac{1}{N} \sum_{i=1}^{N} [y_i * (log y_i - x_i)]$$

&emsp;&emsp;注意这里$x_i$是log 概率


```python
loss_fn = nn.KLDivLoss()
test_input = torch.randn(3,5)
target = torch.randn(3,5)
print('input:' ,test_input)
print('target:',target)
output = loss_fn(test_input,target)
print('output:',output)
```

    input: tensor([[ 0.5973,  0.0002, -0.2455,  1.0198,  1.2702],
            [-1.0674,  0.8753, -1.2455,  0.2580,  1.2803],
            [-2.1533,  1.6474,  0.7867, -1.0147,  0.2658]])
    target: tensor([[-0.9287,  1.3780, -0.2501, -0.7478,  1.8697],
            [ 0.3530, -0.5857, -0.1452,  0.1384, -0.4717],
            [-0.3370, -0.3774, -0.9127, -0.3236,  0.9600]])
    output: tensor(-0.0905)


=======================================

#### 目前主要涉及到这7个，还有hinge loss 等等，这里先不扩展了，接下来仔细总结一下熵的问题。

#### 1.如何量化信息?

&emsp;&emsp;在信息论中，认为:

- 非常可能发生的事件信息量要比较少
- 较不可能发生的事件具有更高的信息量
- 独立事件应具有增量的信息，例如，投掷的硬币两次正面朝上传递的信息量，应该是投掷一次硬币正面朝上的信息量的两倍。

&emsp;&emsp;为了满足上面三个性质，定义自信息：

$$I(x) = -log P(x) $$

&emsp;&emsp;自信息只能处理单个的输出，可以使用香浓熵来对整个概率分布中的不确定性总量进行量化。

$$H(x) = E_{x - P}[I(x)] = \sum_{i=1}^{N}P(x_i)I(x_i) = -\sum_{i=1}^{N}P(x_i) log P(x_i)$$

&emsp;&emsp;熵的一些性质：

- 那些接近确定性的分布具有较低的熵
- 那些接近均匀分布的概率分布具有较高的熵

#### 2. KL散度

&emsp;&emsp;KL散度可以用来衡量两个分布的差异，在概率与统计中，经常会将一个复杂的分布用一个简单的近似分布来代替，KL散度可以帮助测量在选择一个近似分布时丢失的信息量。

&emsp;&emsp;假设原概率分布为$P(x)$，近似概率分布为$Q(x)$,则使用KL散度衡量这两个分布的差异。

$$D_{KL}(P||Q) \sum_{i=1}^{N}P(x_i) log \frac{P(x_i)}{Q(x_i)}
= \sum_{i=1}^{N} p(x_i)[log P(x_i) - log Q(x_i)]$$

&emsp;&emsp;KL散度的一些性质：

- KL散度是非负的。
- KL散度为0，当且仅当P和Q在离散型变量的情况下是相同的分布，或者在连续型变量的情况下是‘几乎处处’相同的。
- KL散度不是真的距离，它不是对称的。

#### 3. 交叉熵

&emsp;&emsp;交叉熵也可以用来衡量两个分布的差异。

$$H(P,Q) = -E_{x - P}log Q(x) = -\sum_{i=1}^{N} P(x_i)log Q(x_i)$$

&emsp;&emsp;交叉熵$H(P,Q) = H(P) + D_{KL}(P||Q)$,其中$H(P)$为分布为P的熵，当概率分布$P(x)$确定时,$H(P)$也将被确定,即是一个常数，在这种情况下，交叉熵和KL三度就差一个大小为H(P)的常数。推倒如下：

$$D_{KL}(P||Q) \sum P(x_i)[log P(x) - log Q(x)] 
\\ = \sum_{i=1}^{N} P(x_i) log P(x_i) - \sum_{i=1}^{N} P(x_i) log Q(x_i) 
\\ = -H(P) + H(P,Q)$$

&emsp;&emsp;即$H(P,Q) = H(P) + D_{KL}(P||Q)$

&emsp;&emsp;交叉熵的一些性质：

- 非负
- 和KL散度相同，交叉熵也不具备对称性
- 对同一个分布求交叉熵等价于对其求熵


主要参考如下:

- [【机器学习基础】熵、KL散度、交叉熵](https://github.com/moneyDboat/data_grand)
- [torch官方文档](https://pytorch.org/docs/stable/nn.html#crossentropyloss)
- [pytorch loss function 总结](https://blog.csdn.net/zhangxb35/article/details/72464152)

