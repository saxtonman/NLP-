一、实验介绍
1. 实验内容

在实验3中，我们通过观察感知器来介绍神经网络的基础，感知器是现存最简单的神经网络。感知器的一个历史性的缺点是它不能学习数据中存在的一些非常重要的模式。例如，查看图4-1中绘制的数据点。这相当于非此即彼(XOR)的情况，在这种情况下，决策边界不能是一条直线(也称为线性可分)。在这个例子中，感知器失败了。
![image](https://github.com/saxtonman/NLP-/assets/173620482/425f6f8e-6720-484e-9ec9-9c76706c43a3)



图4-1 XOR数据集中的两个类绘制为圆形和星形。请注意，没有任何一行可以分隔这两个类。
在这一实验中，我们将探索传统上称为前馈网络的神经网络模型，以及两种前馈神经网络:多层感知器和卷积神经网络。多层感知器在结构上扩展了我们在实验3中研究的简单感知器，将多个感知器分组在一个单层，并将多个层叠加在一起。我们稍后将介绍多层感知器，并在“示例:带有多层感知器的姓氏分类”中展示它们在多层分类中的应用。

本实验研究的第二种前馈神经网络，卷积神经网络，在处理数字信号时深受窗口滤波器的启发。通过这种窗口特性，卷积神经网络能够在输入中学习局部化模式，这不仅使其成为计算机视觉的主轴，而且是检测单词和句子等序列数据中的子结构的理想候选。我们在“卷积神经网络”中概述了卷积神经网络，并在“示例:使用CNN对姓氏进行分类”中演示了它们的使用。

在本实验中，多层感知器和卷积神经网络被分组在一起，因为它们都是前馈神经网络，并且与另一类神经网络——递归神经网络(RNNs)形成对比，递归神经网络(RNNs)允许反馈(或循环)，这样每次计算都可以从之前的计算中获得信息。在实验6和实验7中，我们将介绍RNNs以及为什么允许网络结构中的循环是有益的。

在我们介绍这些不同的模型时，需要理解事物如何工作的一个有用方法是在计算数据张量时注意它们的大小和形状。每种类型的神经网络层对它所计算的数据张量的大小和形状都有特定的影响，理解这种影响可以极大地有助于对这些模型的深入理解。

2. 实验要点

通过“示例:带有多层感知器的姓氏分类”，掌握多层感知器在多层分类中的应用
掌握每种类型的神经网络层对它所计算的数据张量的大小和形状的影响
3. 实验环境

Python 3.6.7
4. 附件目录

请将本实验所需数据文件(surnames.csv)上传至目录：/data/surnames/.
示例完整代码：
exp4-In-Text-Examples.ipynb
exp4-munging_surname_dataset.ipynb
exp4-2D-Perceptron-MLP.ipynb
exp4_4_Classify_Surnames_CNN.ipynb
exp4_4_Classify_Surnames_MLP.ipynb
二、The Multilayer Perceptron（多层感知器）
多层感知器(MLP)被认为是最基本的神经网络构建模块之一。最简单的MLP是对第3章感知器的扩展。感知器将数据向量作为输入，计算出一个输出值。在MLP中，许多感知器被分组，以便单个层的输出是一个新的向量，而不是单个输出值。在PyTorch中，正如您稍后将看到的，这只需设置线性层中的输出特性的数量即可完成。MLP的另一个方面是，它将多个层与每个层之间的非线性结合在一起。

最简单的MLP，如图4-2所示，由三个表示阶段和两个线性层组成。第一阶段是输入向量。这是给定给模型的向量。在“示例:对餐馆评论的情绪进行分类”中，输入向量是Yelp评论的一个收缩的one-hot表示。给定输入向量，第一个线性层计算一个隐藏向量——表示的第二阶段。隐藏向量之所以这样被调用，是因为它是位于输入和输出之间的层的输出。我们所说的“层的输出”是什么意思?理解这个的一种方法是隐藏向量中的值是组成该层的不同感知器的输出。使用这个隐藏的向量，第二个线性层计算一个输出向量。在像Yelp评论分类这样的二进制任务中，输出向量仍然可以是1。在多类设置中，将在本实验后面的“示例:带有多层感知器的姓氏分类”一节中看到，输出向量是类数量的大小。虽然在这个例子中，我们只展示了一个隐藏的向量，但是有可能有多个中间阶段，每个阶段产生自己的隐藏向量。最终的隐藏向量总是通过线性层和非线性的组合映射到输出向量。

![image](https://github.com/saxtonman/NLP-/assets/173620482/0db42e6c-0ea6-410e-b837-7078674ab931)
图4-2 一种具有两个线性层和三个表示阶段（输入向量、隐藏向量和输出向量)的MLP的可视化表示
mlp的力量来自于添加第二个线性层和允许模型学习一个线性分割的的中间表示——该属性的能表示一个直线(或更一般的,一个超平面)可以用来区分数据点落在线(或超平面)的哪一边的。学习具有特定属性的中间表示，如分类任务是线性可分的，这是使用神经网络的最深刻后果之一，也是其建模能力的精髓。在下一节中，我们将更深入地研究这意味着什么。

2.1 A Simple Example: XOR

让我们看一下前面描述的XOR示例，看看感知器与MLP之间会发生什么。在这个例子中，我们在一个二元分类任务中训练感知器和MLP:星和圆。每个数据点是一个二维坐标。在不深入研究实现细节的情况下，最终的模型预测如图4-3所示。在这个图中，错误分类的数据点用黑色填充，而正确分类的数据点没有填充。在左边的面板中，从填充的形状可以看出，感知器在学习一个可以将星星和圆分开的决策边界方面有困难。然而，MLP(右面板)学习了一个更精确地对恒星和圆进行分类的决策边界。
![image](https://github.com/saxtonman/NLP-/assets/173620482/5e646ce5-9d37-4067-9538-dbdaf26ca384)
图4-3 从感知器(左)和MLP(右)学习的XOR问题的解决方案显示
图4-3中，每个数据点的真正类是该点的形状:星形或圆形。错误的分类用块填充，正确的分类没有填充。这些线是每个模型的决策边界。在边的面板中，感知器学习—个不能正确地将圆与星分开的决策边界。事实上，没有一条线可以。在右动的面板中，MLP学会了从圆中分离星。

虽然在图中显示MLP有两个决策边界，这是它的优点，但它实际上只是一个决策边界!决策边界就是这样出现的，因为中间表示法改变了空间，使一个超平面同时出现在这两个位置上。在图4-4中，我们可以看到MLP计算的中间值。这些点的形状表示类(星形或圆形)。我们所看到的是，神经网络(本例中为MLP)已经学会了“扭曲”数据所处的空间，以便在数据通过最后一层时，用一线来分割它们。
![image](https://github.com/saxtonman/NLP-/assets/173620482/a044764c-5870-405b-bbb3-905f3958249b)
图4-4 MLP的输入和中间表示是可视化的。从左到右:（1）网络的输入;（2）第一个线性模块的输出;（3）第一个非线性模块的输出;（4）第二个线性模块的输出。第一个线性模块的输出将圆和星分组，而第二个线性模块的输出将数据点重新组织为线性可分的。
相反，如图4-5所示，感知器没有额外的一层来处理数据的形状，直到数据变成线性可分的。
![image](https://github.com/saxtonman/NLP-/assets/173620482/cd5c447b-192a-4fbb-bda4-fc34b7a0019f)
图4-5 感知器的输入和输出表示。因为它没有像MLP那样的中间表示来分组和重新组织，所以它不能将圆和星分开。
2.2 Implementing MLPs in PyTorch

在上一节中，我们概述了MLP的核心思想。在本节中，我们将介绍PyTorch中的一个实现。如前所述，MLP除了实验3中简单的感知器之外，还有一个额外的计算层。在我们在例4-1中给出的实现中，我们用PyTorch的两个线性模块实例化了这个想法。线性对象被命名为fc1和fc2，它们遵循一个通用约定，即将线性模块称为“完全连接层”，简称为“fc层”。除了这两个线性层外，还有一个修正的线性单元(ReLU)非线性(在实验3“激活函数”一节中介绍)，它在被输入到第二个线性层之前应用于第一个线性层的输出。由于层的顺序性，必须确保层中的输出数量等于下一层的输入数量。使用两个线性层之间的非线性是必要的，因为没有它，两个线性层在数学上等价于一个线性层4，因此不能建模复杂的模式。MLP的实现只实现反向传播的前向传递。这是因为PyTorch根据模型
的定义和向前传递的实现，自动计算出如何进行向后传递和梯度更新。

Example 4-1. Multilayer Perceptron
import torch.nn as nn
import torch.nn.functional as F

class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): the size of the input vectors
            hidden_dim (int): the output size of the first Linear layer
            output_dim (int): the output size of the second Linear layer
        """
        super(MultilayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """The forward pass of the MLP

        Args:
            x_in (torch.Tensor): an input data tensor.
                x_in.shape should be (batch, input_dim)
            apply_softmax (bool): a flag for the softmax activation
                should be false if used with the Cross Entropy losses
        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)

        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output

在例4-2中，我们实例化了MLP。由于MLP实现的通用性，可以为任何大小的输入建模。为了演示，我们使用大小为3的输入维度、大小为4的输出维度和大小为100的隐藏维度。请注意，在print语句的输出中，每个层中的单元数很好地排列在一起，以便为维度3的输入生成维度4的输出。

Example 4-2. An example instantiation of an MLP


batch_size = 2 # number of samples input at once
input_dim = 3
hidden_dim = 100
output_dim = 4

# Initialize model
mlp = MultilayerPerceptron(input_dim, hidden_dim, output_dim)
print(mlp)

MultilayerPerceptron(
  (fc1): Linear(in_features=3, out_features=100, bias=True)
  (fc2): Linear(in_features=100, out_features=4, bias=True)
)

我们可以通过传递一些随机输入来快速测试模型的“连接”，如示例4-3所示。因为模型还没有经过训练，所以输出是随机的。在花费时间训练模型之前，这样做是一个有用的完整性检查。请注意PyTorch的交互性是如何让我们在开发过程中实时完成所有这些工作的，这与使用NumPy或panda没有太大区别:

Example 4-3. Testing the MLP with random inputs

import torch
def describe(x):
    print("Type: {}".format(x.type()))
    print("Shape/size: {}".format(x.shape))
    print("Values: \n{}".format(x))

x_input = torch.rand(batch_size, input_dim)
describe(x_input)

Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values: 
tensor([[0.2969, 0.4169, 0.4621],
        [0.4702, 0.0020, 0.6063]])
上述代码运行结果：

Type: torch.FloatTensor
Shape/size: torch.Size([2, 3])
Values: 
tensor([[0.6193, 0.7045, 0.7812],
        [0.6345, 0.4476, 0.9909]])
y_output = mlp(x_input, apply_softmax=False)
describe(y_output)
Type: torch.FloatTensor
Shape/size: torch.Size([2, 4])
Values: 
tensor([[-0.1551, -0.1124, -0.2134, -0.0665],
        [-0.0969, -0.1321, -0.3118, -0.0283]], grad_fn=<AddmmBackward>)
上述代码运行结果：

Type: torch.FloatTensor
Shape/size: torch.Size([2, 4])
Values: 
tensor([[ 0.2356,  0.0983, -0.0111, -0.0156],
        [ 0.1604,  0.1586, -0.0642,  0.0010]], grad_fn=<AddmmBackward>)


学习如何读取PyTorch模型的输入和输出非常重要。在前面的例子中，MLP模型的输出是一个有两行四列的张量。这个张量中的行与批处理维数对应，批处理维数是小批处理中的数据点的数量。列是每个数据点的最终特征向量。在某些情况下，例如在分类设置中，特征向量是一个预测向量。名称为“预测向量”表示它对应于一个概率分布。预测向量会发生什么取决于我们当前是在进行训练还是在执行推理。在训练期间，输出按原样使用，带有一个损失函数和目标类标签的表示。我们将在“示例:带有多层感知器的姓氏分类”中对此进行深入介绍。

但是，如果想将预测向量转换为概率，则需要额外的步骤。具体来说，需要softmax函数，它用于将一个值向量转换为概率。softmax有许多根。在物理学中，它被称为玻尔兹曼或吉布斯分布;在统计学中，它是多项式逻辑回归;在自然语言处理(NLP)社区，它是最大熵(MaxEnt)分类器。不管叫什么名字，这个函数背后的直觉是，大的正值会导致更高的概率，小的负值会导致更小的概率。在示例4-3中，apply_softmax参数应用了这个额外的步骤。在例4-4中，可以看到相同的输出，但是这次将apply_softmax标志设置为True:

Example 4-4. MLP with apply_softmax=True

y_output = mlp(x_input, apply_softmax=True)
describe(y_output)
Type: torch.FloatTensor
Shape/size: torch.Size([2, 4])
Values: 
tensor([[0.2451, 0.2558, 0.2312, 0.2678],
        [0.2602, 0.2512, 0.2099, 0.2787]], grad_fn=<SoftmaxBackward>)
上述代码运行结果：

Type: torch.FloatTensor
Shape/size: torch.Size([2, 4])
Values: 
tensor([[0.2915, 0.2541, 0.2277, 0.2267],
        [0.2740, 0.2735, 0.2189, 0.2336]], grad_fn=<SoftmaxBackward>)
综上所述，mlp是将张量映射到其他张量的线性层。在每一对线性层之间使用非线性来打破线性关系，并允许模型扭曲向量空间。在分类设置中，这种扭曲应该导致类之间的线性可分性。另外，可以使用softmax函数将MLP输出解释为概率，但是不应该将softmax与特定的损失函数一起使用，因为底层实现可以利用高级数学/计算捷径。

三、实验步骤
3.1 Example: Surname Classification with a Multilayer Perceptron

在本节中，我们将MLP应用于将姓氏分类到其原籍国的任务。从公开观察到的数据推断人口统计信息(如国籍)具有从产品推荐到确保不同人口统计用户获得公平结果的应用。人口统计和其他自我识别信息统称为“受保护属性”。“在建模和产品中使用这些属性时，必须小心。”我们首先对每个姓氏的字符进行拆分，并像对待“示例:将餐馆评论的情绪分类”中的单词一样对待它们。除了数据上的差异，字符层模型在结构和实现上与基于单词的模型基本相似.

应该从这个例子中吸取的一个重要教训是，MLP的实现和训练是从我们在第3章中看到的感知器的实现和培训直接发展而来的。事实上，我们在实验3中提到了这个例子，以便更全面地了解这些组件。此外，我们不包括“例子:餐馆评论的情绪分类”中看到的代码。

本节的其余部分将从姓氏数据集及其预处理步骤的描述开始。然后，我们使用词汇表、向量化器和DataLoader类逐步完成从姓氏字符串到向量化小批处理的管道。如果你通读了实验3，应该知道，这里只是做了一些小小的修改。

我们将通过描述姓氏分类器模型及其设计背后的思想过程来继续本节。MLP类似于我们在实验3中看到的感知器例子，但是除了模型的改变，我们在这个例子中引入了多类输出及其对应的损失函数。在描述了模型之后，我们完成了训练例程。训练程序与“示例:对餐馆评论的情绪进行分类”非常相似，因此为了简洁起见，我们在这里不像在该部分中那样深入，可以回顾这一节内容。

3.1.1 The Surname Dataset

姓氏数据集，它收集了来自18个不同国家的10,000个姓氏，这些姓氏是作者从互联网上不同的姓名来源收集的。该数据集将在本课程实验的几个示例中重用，并具有一些使其有趣的属性。第一个性质是它是相当不平衡的。排名前三的课程占数据的60%以上:27%是英语，21%是俄语，14%是阿拉伯语。剩下的15个民族的频率也在下降——这也是语言特有的特性。第二个特点是，在国籍和姓氏正字法(拼写)之间有一种有效和直观的关系。有些拼写变体与原籍国联系非常紧密(比如“O ‘Neill”、“Antonopoulos”、“Nagasawa”或“Zhu”)。

为了创建最终的数据集，我们从一个比课程补充材料中包含的版本处理更少的版本开始，并执行了几个数据集修改操作。第一个目的是减少这种不平衡——原始数据集中70%以上是俄文，这可能是由于抽样偏差或俄文姓氏的增多。为此，我们通过选择标记为俄语的姓氏的随机子集对这个过度代表的类进行子样本。接下来，我们根据国籍对数据集进行分组，并将数据集分为三个部分:70%到训练数据集，15%到验证数据集，最后15%到测试数据集，以便跨这些部分的类标签分布具有可比性。

SurnameDataset的实现与“Example: classification of Sentiment of Restaurant Reviews”中的ReviewDataset几乎相同，只是在getitem方法的实现方式上略有不同。回想一下，本课程中呈现的数据集类继承自PyTorch的数据集类，因此，我们需要实现两个函数:__getitem方法，它在给定索引时返回一个数据点;以及len方法，该方法返回数据集的长度。“示例:餐厅评论的情绪分类”中的示例与本示例的区别在getitem__中，如示例4-5所示。它不像“示例:将餐馆评论的情绪分类”那样返回一个向量化的评论，而是返回一个向量化的姓氏和与其国籍相对应的索引:

Example 4-5. Implementing SurnameDataset.__getitem__()

class SurnameDataset(Dataset):
    # Implementation is nearly identical to Section 3.5

    def __getitem__(self, index):
        row = self._target_df.iloc[index]
        surname_vector = \
            self._vectorizer.vectorize(row.surname)
        nationality_index = \
            self._vectorizer.nationality_vocab.lookup_token(row.nationality)

        return {'x_surname': surname_vector,
                'y_nationality': nationality_index}




