---
layout: post
title: "Texture Matching for Automatic Slice Verification"
author: "jokoe66"
header-style: text
tags:
  - image processing
---

### Introduction

滑块验证码是用于反爬虫的一种行为验证方法。它通过让用户移动滑块完成拼图来进行行为验证。

<img src="/img/in-post/CV/slice_verification/example.png" width=250px>

<center>图1. 滑动验证码示例。</center> 

破解这种验证方式的思路一般是先识别缺口（豁口）的位置，然后用自动化工具（如 selenium）将滑块滑动到缺口位置。

本文提出了基于纹理匹配的缺口识别算法。我们的直觉是缺口处的纹理和滑块的纹理是相似的。实验结果验证了我们的直觉和算法的高准确性和鲁棒性。

![method_example.png](/img/in-post/CV/slice_verification/method_example.png)

<center>图2. 基于纹理匹配的缺口识别算法。</center> 

### Texture Matching

对于输入的滑块图像和带缺口的背景图像，我们在背景图像上搜索与滑块图像的纹理最相似的区域位置。首先提取滑块和带缺口背景图的边缘纹理特征，然后在背景纹理图的每个位置上进行局部纹理与滑块纹理的相关度计算，最后取相关度最高的位置作为最匹配位置。

纹理特征使用经典的 Canny 边缘描述子来提取。

相关度为归一化的相关度系数，可以看作展平后的一维图像向量的余弦相似度，计算公式如下：

![ccoeff_normed.png](/img/in-post/CV/slice_verification/ccoeff_normed.png)

其中 T' 为去均值的滑块纹理图像矩阵， I' 为去均值的背景纹理图像矩阵

![ccoeff_normed_appendix.png](/img/in-post/CV/slice_verification/ccoeff_normed_appendix.png)

代码实现如下

```python
import cv2

def detect_fade(bg_with_fade, slice_block):
    """ 缺口检测算法

    Args:
        bg_with_fade(numpy.ndarray): 带缺口的背景图 (h, w, 3)
        slice_block(numpy.ndarray): 滑块图 (h', w', 3)
    Return:
        tuple: 缺口边框 (top, left, bottom, right)
    """
    # Edge detection
    img = cv2.Canny(bg_with_fade, 100, 200)
    template = cv2.Canny(slice_block, 100, 200)
    w, h = template.shape[::-1]
    # Apply template Matching
    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    # Find most matched position
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    return top_left + bottom_right
```

### Results

#### Dataset

验证码图像数据从”极验验证码“官网爬取，每个样本是由滑块图和带缺口的背景图构成的二元组。

<div align=center>
  <div style="display:inline-block">
    <img src="/img/in-post/CV/slice_verification/slice_block.png" width=150px>
    <center> 图3.(a) 滑块图示例。</center>
  </div>
  <div style="display:inline-block">
    <img src="/img/in-post/CV/slice_verification/bg_with_fade.png" width=150px>
    <center> 图3.(b) 背景图示例。</center>
  </div>
</div>

#### Accuracy

![demo1.png](/img/in-post/CV/slice_verification/demo1.png)

![demo2.png](/img/in-post/CV/slice_verification/demo2.png)

![demo3.png](/img/in-post/CV/slice_verification/demo3.png)

![demo4.png](/img/in-post/CV/slice_verification/demo4.png)

![demo5.png](/img/in-post/CV/slice_verification/demo5.png)

![demo6.png](/img/in-post/CV/slice_verification/demo6.png)

![demo7.png](/img/in-post/CV/slice_verification/demo7.png)

![demo8.png](/img/in-post/CV/slice_verification/demo8.png)

![demo9.png](/img/in-post/CV/slice_verification/demo9.png)

####  Speed

|      CPU       | resolution | fps  |
| :------------: | :--------: | :--: |
| Intel i5 8665u | 260 x 160  | 515  |

### Conclusion

本文提出了基于纹理匹配的滑块验证码缺口识别算法，该算法在背景图上搜索与滑块纹理相似的区域。从实验结果来看，该算法不仅准确率高而且速度快。