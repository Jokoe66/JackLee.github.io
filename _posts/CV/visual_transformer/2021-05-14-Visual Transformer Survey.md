---
layout: post
title: "Visual Transformer Survey"
author: "jokoe66"
header-style: text
tags:
  - cv
  - transformer
  - attention machenism
---

**1. 简介**

transformer 在 nlp 领域获得了极大的成功，许多研究者开始探究它在 cv 领域的应用前景。本文总结了图像分类、检测等研究方向上的基于 transformer 的研究，旨在回答：

1. transformer 相比于传统的基于卷积神经网络的视觉模型有什么样的优劣？
2. transformer 是怎样应用于图像任务上的？

**1.1 Transformer**

transformer 是一种基于 self-attention 的序列转换模型，具有很强的序列模式提取的能力。

在 transformer 之前，主流的序列模型是包含复杂的卷积和循环神经网络的 encoder - decoder 模型。这种模型的主要缺陷在于难以学习长距离的依赖关系，因为某个位置的输入信号需要经过若干次循环神经网络才能到达上下文所在位置。除了长距离建模能力的缺陷以外，传统模型的并行计算能力很差，计算速度很慢。针对这些问题，google brain 提出了基于 attention 的 transformer。

**Encoder - decoder**

transformer 整体上是一种 encoder - decoder 结构。encoder 输出是与输入序列长度相同的融合了上下文信息的序列，decoder 每一步预测一个位置的输出从而得到一个输出序列。decoder 是自回归的，每次预测都会以之前的预测作为额外的输入。

![transformer_arch.png](/img/in-post/CV/visual_transformer/transformer_arch.png)

encoder 由多个相同的网络层堆叠而成，每个网络层有两个子层，分别是多头注意力层和前馈网络层。这两个子层都使用了残差连接和 layer normalization。

decoder 同样由多个相同的网络层堆叠而成，不同于 encoder 的是，每个网络层除了多头注意力和前馈网络以外，还增加了对 encoder 输出使用多头注意力的第三个子层。为了防止 decoder 每次预测关注到当前位置之后的无用输入，decoder 的多头注意力使用 mask 使得每一步的预测只与当前位置之前的序列输入有关。

**Multi-head Attention**

多头注意力由多个相同、并行的注意力机制组成。注意力机制是对输入序列进行加权平均的操作，而权重是与输入序列有关的注意力。输入序列首先通过线性变换映射到 query, key 和 value 三个维度相同的空间中，注意力权重由 query 和 key 的归一化内积得到，输出由注意力加权的平均 value。

![multi-head-attention.png](/img/in-post/CV/visual_transformer/multi-head-attention.png)

**Position Encoding**

在不使用循环或者卷积神经网络时，transformer 无法感知输入序列中元素的顺序关系。为了注入序列中每个元素的位置信息，transformer 使用了 position encoding。position encoding 有多种方式，例如直接使用元素的绝对或相对索引，但这两种方式前者存在 position encoding 值过大的问题，后者存在长度不同的文本中距离相同的两个元素的 position encoding 差值不同的问题。

另一种方案是使用绝对位置的二进制编码。

![position-encoding-binary.png](/img/in-post/CV/visual_transformer/position-encoding-binary.png)

transformer 使用了不同频率的正余弦函数的组合来做位置编码。使用正余弦函数的直觉与使用二进制数字类似，二进制数字的不同比特位对应了不同频率的正余弦函数。因此这种编码方式是可以表示位置信息的。

位置 t 的位置编码 pt 的维度为 d，

![position-encoding-eq1.png](/img/in-post/CV/visual_transformer/position-encoding-eq1.png)

每个位置编码由若干个不同频率的正余弦函数值对构成，频率随着维度的增加而逐渐降低。

![position-encoding-eq2.png](/img/in-post/CV/visual_transformer/position-encoding-eq2.png)

![position-encoding-eq3.png](/img/in-post/CV/visual_transformer/position-encoding-eq3.png)

使用正余弦函数做位置编码的另一个好处是容易使模型利用到相对位置的信息。对于任意固定距离的两个位置，它们的位置编码是线性的关系，它们的相对位置编码可以通过位置编码简单的线性组合来得到。

这个性质有数学证明，而直观理解是把位置编码中的不同频率的正余弦函数对看成时钟上的速度不同指针（时针、分针），对于每个指针，从一个位置移动到另一个位置一定可以通过一个旋转线性变换来实现。

**1.2 Visual Transformers**

​		transformer 在视觉领域的研究可以分为辅助 CNN 和完全替代 CNN 两类。其中辅助 CNN 的方法在现阶段更为成熟、实际落地效果更好。辅助 CNN 的方法是将 self-attention 或 transformer 用于 CNN 提取的抽象特征上，用于增强抽象概念之间的远距离依赖关系的提取能力。完全替代 CNN 的方法将 self-attention 作为模型的基本组成部分，完成包括浅层图像处理、抽象特征提取的整个图像理解任务。

**1.2.1 Stand-alone Image Recognition with Self-Attention**

* **An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale**

  **—— ICLR 2021 Google Brain 2020.10**

  这篇 paper 研究了直接使用原始的 transformer 进行图像内容理解。目前 transformer 在 cv 领域的应用要么是用于辅助 CNN 提取长距离依赖，要么是应用于局部特征提取。作者彻底摒弃了卷积操作，将 transformer 直接用于图像内容理解。在训练集足够大时，这种方法的准确率超过了基于卷积神经网络的方法，而具有更高的计算效率。

* **On the Relationship between Self-Attention and Convolutional Layers**

  **—— ICLR 2020 洛桑理工 2019.11**

  这篇 paper 研究了卷积和 self-attention 操作的关系。已有的研究表明单独由 self-attention 构成的模型也能达到与深度卷积神经网络模型相当的图像分类效果。那么 self - attention 与卷积操作之间是什么样的关系？这篇 paper 证明了 self-attention 也具有卷积的表达能力、并且实际上 self-attention 在图像处理上完成了与卷积类似的功能。多头注意力机制具有至少与卷积相当的表达能力，数值实验表明 self-attention 处理像素网格中的视觉模式的方式与卷积类似。

* *** Stand-Alone Self-Attention in Vision Models**

  **—— NIPS 2019 Google Brain 2019.6**

  **1. Introduction**

  这篇 paper 研究了如何使用 self-attention 完全替代卷积进行图像识别。当时主流的 self-attention 在 cv 上的应用是用 self-attention 辅助增强卷积的长距离依赖关系提取能力。作者设计了完全基于 self-attention 的模型验证了 self-attention 可以作为一个独立的网络层。作者将 CNN 中所有的空间卷积操作用特定形式的 self-attention 替代，得到的图像分类和目标检测网络效果超过了 baseline。

  **2. Fully Attentional Vision Models**

  以 self-attention 作为基本单元来构建完全视觉模型包括替换卷积模块和替换第一个卷积操作。

  **2.1 Replacing Spatial Convolutions**

  空间卷积指的是卷积核尺寸大于 1 的卷积。作者使用了直接的策略，将卷积操作在每个空间位置的内积用多头 self-attention 替代。

  ![local-attention.png](/img/in-post/CV/visual_transformer/local-attention.png)

  在每个空间位置，作者将通道分成 N 组，在每个组内做 self-attention，之后将 N 组 self-attention 的结果拼接起来，得到 multi-head self-attention 的结果。

  为了引入位置信息，作者使用相对位置 embedding 作为位置编码。相对位置指的是像素之间的位置偏移，包括行偏移和列偏移，二者的拼接作为最终的位置编码。行偏移和列偏移的 embedding 是可学习的。引入位置编码的 self-attention 如下所示。

  ![/img/in-post/CV/visual_transformer/local-attention-with-pos-encoding.png](/img/in-post/CV/visual_transformer/local-attention-with-pos-encoding.png)

  **2.2 Replacing the Convolutional Stem**

  convolutional stem 指的是 CNN 的第一个卷积操作，它一般用于从原始 RGB 图像中提取浅层特征。将它直接用 self-attention 替换是不可取的，因为 RGB 像素包含的信息量很少，无法从中获取有效的 attention。这点有大量的实验支撑。直接替换为 self-attention 的话模型的性能会特别差。

  为了高效地获取对提取高层特征有用的边缘、局部特征，作者引入基于距离的卷积核参数化。基于这种卷积提取的特征做 self-attention 的效果可以大大提升。

  ![attention-stem.png](/img/in-post/CV/visual_transformer/attention-stem.png)

* **Local Relation Networks for Image Recognition**

  **—— ICCV 2019 Microsoft 2019.4**

  这篇 paper 设计了基于 self-attention 的特征提取方法。在卷积特征提取中，空间聚合本质上是一个使用固定的卷积核来进行模式匹配的过程，这种方式在建模空间分布多变的视觉元素上效率低下。作者提出基于 self-attention 的特征提取网络 local relation layer 来增强空间聚合的能力。这种特征提取方法可以根据像素之间的关系来调整空间聚合时不同像素的权重，灵活地将不同像素组合成更高级的实体。作者用这种特征提取方法替代卷积的空间聚合，设计了完全基于 self-attention 的 LR - Net，其分类效果超过了基于卷积空间聚合的方法。

**1.2.2 Combining Self-Attention with CNN**

* *** Visual Transformers: Token-based Image Representation and Processing for Computer Vision**

  **—— FAIR 2020.6**

  **1. Introduction**

  这篇 paper 提出用 transformer 增强 CNN 提取的高层概念，结合了 CNN 和 transformer 二者的优点，简单实用。目前主流的图像建模方式是以规则的像素序列来表示图像，以卷积的形式提取各个空间位置所有可能的抽象概念。这种建模方式的主要缺点是：1）有限的感受野使得难以有效建模长距离的依赖；2）以固定模板匹配的方式提取特征的方式会浪费很多的参数和计算量。为了解决这些问题，Visual Transformer 提出用视觉符号表示图像、用 transformer 来关联远距离的抽象概念。在图像分类、场景分割等多个任务上，Visual Transformer 的准确率明显高于仅使用 CNN 的方法。

  **2. Visual Transformer**

  基于 Visual Transformer 的模型将输入图像分别用 CNN 和 VTs 依次处理。这种做法结合了 CNN 和 Transformer 的优势：首先将输入图像通过 CNN 提取空间上分布密集的浅层特征，然后通过 VTs 提取空间上分布稀疏的更高层的语义概念。最后模型基于语义概念进行图像分类或基于融合了语义概念的特征图进行图像分割。

  VTs 包括将像素按语义聚合为若干视觉符号、使用 Transformer 关联视觉符号和将视觉符号投影到特征图上来增强特征图三步。这个过程与先前的一些工作类似，不同之处在于 VTs 不需要数百个视觉符号，仅需要少量（16个）视觉符号就可以取得很好的性能。

  **2.1 Tokenizer**

  引入视觉符号的直觉是一张图像可以用来一些词（或者说视觉符号）来描述。为了提取视觉符号，作者设计了 Tokenizer，将特征图转换为视觉符号集合。Tokenizer 的输入是形状为 H x W x C 的特征图 X，输出形状为 L x C 的视觉符号 T。

  **2.1.1 Filter-based Tokenizer**

  Tokenizer 的一种实现是使用基于卷积滤波器来提取视觉符号。特征图经过 1x1 卷积得到形状为 H x W x L 的 attention map，表示每个视觉符号对输入特征图的空间注意力，这个 attention map 与输入特征图相乘进行加权平均得到 L 个维度为 C 的视觉符号。

  这种固定的卷积核可能会因为直接对所有高层概念进行建模而产生潜在的计算浪费。

  **2.1.2 Recurrent Tokenizer**

  为了提高 Tokenizer 的效率，作者引入了 recurrent Tokenizer。recurrent tokenizer 的结构与 filter-based tokenizer 大致一样，仅卷积核不同。它的卷积核是由上一个 VTs 提取的视觉符号经过线性变换得到的。这种基于内容的特征提取方式实际上与 self-attention 类似，上一个 VTs 提取的视觉符号经过线性变换得到 query，query 通过卷积的方式与作为 key 的特征图进行内积运算，得到 attention map，最后通过对作为 value 的特征图进行加权平均得到新的视觉符号。

  ![recurrent-tokenizer.png](/img/in-post/CV/visual_transformer/recurrent-tokenizer.png)

  **2.2 Transformer**

  在提取视觉符号之后，作者使用 Transformer对视觉符号之间的交互进行建模。Transformer 由标准的 self-attention 组成。

  **2.3 Projector**

  与 tokenizer 类似的，视觉符号可以通过 projector 投影回特征图上。Projector 也是基于 attention 完成的。

* **End-to-end object detection with transformers**

  **—— ECCV 2020 FAIR**

  这篇 paper 将 transformer 用在 CNN 特征上进行目标检测任务。

* **Attention augmented convolutional networks**

  **—— ICCV 2019 Google Brain**

  这篇 paper 提出用 attention 提取全局特征来增强卷积特征。作者探究了 attention 在图像识别任务中的作用，发现将 attention 与卷积结合起来效果是最好的，于是用 attention 来增强卷积操作，将 attention 提取的特征图与卷积特征图拼接起来。
