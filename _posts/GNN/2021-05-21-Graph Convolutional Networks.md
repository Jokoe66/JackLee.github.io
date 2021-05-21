---
layout: post
title: "Semi-Supervised Classification With Graph Convolutional Networks"
author: "jokoe66"
header-style: text
tags:
  - machine learning
  - gnn
  - gcn
  - spectral graph convolution
---

### Introduction

这篇 paper 提出一种高效的 CNN 变种（GCN）对图结构数据进行半监督学习。GCN 的设计基于谱图卷积的局部一阶近似。这种模型的计算量与边的数量成线性关系，学习到的隐藏层表征编码了局部图结构和节点的特征。GCN 在各种图结构数据半监督学习任务上都取得了优于其他 SOTA 方法的结果。

### Convolution on Graph

卷积操作在结构规则的数据（如序列、图像）上的定义是很清晰的，而在图、流形等不规则数据上由于无法明确上下左右等，卷积操作无法直接应用。[David K Hammond](https://arxiv.org/abs/0912.3848) 等人利用谱图理论将卷积操作映射到谱空间的乘积操作，从而实现了谱图卷积（spectral graph convolution)。


$$
g_{\theta} \star x = Ug_{\theta}U^Tx
$$


这里 $g_{\theta}$ 为参数化的卷积核，$g_{\theta}$ 为对角元素为 $\theta \in R^N$ 的对角矩阵，$U$ 和 $U^T$ 分别为谱反变换和谱变换矩阵。

谱图卷积有若干限制：

1）求解谱变换矩阵和谱变换的过程的计算量大，难以应用于大型的图数据上；

2）参数量大，与图顶点数呈线性关系。

为了提高谱图卷积的效率，[Hammond et.al.](https://arxiv.org/abs/0912.3848) 和 [Defferrard et.al](https://arxiv.org/abs/1606.09375) 使用切比雪夫多项式近似 $g_\theta$ 。


$$
g_{\theta'}(\Lambda) \approx \sum_{k=0}^{K}\theta^{'}_k T_k(\hat \Lambda)
$$


这里 $\Lambda$ 为图拉普拉斯的特征值构成的对角矩阵，$\hat \Lambda$ 为调整后的 $\Lambda$， $\hat \Lambda = \frac{2}{\lambda_{max}}\Lambda - I_N$ （$\lambda_{max}$ 为图拉普拉斯的最大特征值），$T_k$ 为切比雪夫多项式，$\theta^{\prime}_k$ 为切比雪夫多项式系数。

由于 $(U\Lambda U^T)^k = U\Lambda ^k U^T$ ，谱图卷积可以近似为


$$
g_{\theta^{\prime}} \star x \approx \sum_{k=0}^{K}\theta^{\prime}_k T_k(\hat L)x
$$


这里 $\hat L$ 为调整后的图拉普拉斯，$\hat L = \frac{2}{\lambda_{max}}L - I_N$ ，$L = I_N - D^{-\frac{1}{2}}AD^{-\frac{1}{2}}$，$D_{ii} = \sum_j A_{ij}$ 为度矩阵。

由于 $L$ 的稀疏性，这个算式的计算复杂度是 $O(\|E\|)$ ，即与图的边数呈线性关系。

GCN 进一步用一阶切比雪夫多项式（$K=1$）以及 $\lambda_{max} \approx 2$ 进行近似


$$
g_{\theta^{\prime}} \star x \approx \theta^{\prime}_0 x + \theta^{\prime}_1 (L - I_N) x = \theta^{\prime}_0 x - \theta^{\prime}_1 D^{-\frac{1}{2}}AD^{-\frac{1}{2}}x
$$


考虑到使用少量参数可以有效防止过拟合，作者仅保留一个参数 $\theta = \theta^{\prime}_0 = -\theta^{\prime}_1 $  ，得到


$$
g_{\theta'} \star x \approx  \theta (I_N + D^{-\frac{1}{2}}AD^{-\frac{1}{2}})x
$$


考虑到 $I_N + D^{-\frac{1}{2}} AD^{-\frac{1}{2}} $ 的特征值大小在 $[0, 2]$ 区间内，当在深度模型中重复使用这种操作时可能导致数值不稳定，作者使用了 *renormalization trick*：$I_N + D^{-\frac{1}{2}} AD^{-\frac{1}{2}} \rightarrow {\hat D}^{-\frac{1}{2}} {\hat A}{\hat D}^{-\frac{1}{2}}$ ，${\hat A} = A + I_N$，$\hat D_i = \Sigma_j \hat A_{ij}$ 。

将这种定义推广到多特征数据 $X \in R^{N \times C}$ ，有


$$
Z = {\hat D}^{-\frac{1}{2}}{\hat A}{\hat D}^{-\frac{1}{2}}X\Theta
$$


$\Theta \in R^{C \times F}$ ，$Z \in R^{N \times F}$。

### Semi-Supervised Node Classification

图半监督学习指的是只有一部分顶点有标签的顶点预测任务，例如引用网络中的文档分类。这篇 paper 提出用 GCN 编码图结构信息并使用所有带标签的顶点进行训练。这样梯度会沿着图的边传递到各个顶点上，使得有标签和无标签的顶点表征都能得到训练。

GCN 是逐层应用谱图卷积的神经网络，如 2 层 GCN 表示如下：


$$
Z = f(X, A) = softmax({\hat A}~ReLU({\hat A} XW^{(0)})~ W^{(1)})
$$


GCN 的参数使用交叉熵损失进行学习


$$
L = - \sum_{l \in Y_L} \sum_{f=1}^{F} Y_{lf} \ln Z_{lf}
$$
