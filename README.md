# 介绍
基于原作者gd2016229035参加2018天池FashionAI服装属性识别竞赛的开源代码，我做了一定修改以适配我的应用场景。
所有"my_"开头的py文件和bat文件都是我修改后的内容。

主要修改包括：
1. 适配Windows 10下，python3.7 + mxnet1.7 + cuda10.2 的单GPU（Nvidia GTX 1060 6G显存）学习环境。
2. 重写了预处理的prepare_data.py文件，重写了加载数据和label的customdataset.py文件。重写后的文件前缀了"my_"。
3. 新增一个my_predict.py文件，用于预测和导出模型。
4. 其他一些我训练和预测过程中发现的小bug的修改。


# The following passages are the orginal README. 以下是原作者的README。

## Introduction
This is the main **Gluon** code of [阿里天池竞赛——服饰属性标签识别](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649). Note that this code is just a part of our final code, but provides the one of our **best single model**. Final submission is a ensemble model of two model: One is this resnet152v2 model ,and the other is the Inceptionv4 model from my teammates.
The code is based on [hetong007's code](https://github.com/hetong007/Gluon-FashionAI-Attributes) which provides a good baseline in the competition. This is my first time to use Gluon and thanks to hetong007~

Team name：时尚时尚最时尚
Rank: 10/2950 (Season1)    17/2950 (Season2)


## Software:
- ubuntu14.04，cuda8.0，cudnn6.5
- python2.7
- mxnet-cu80
- numpy
- pandas


## Highlights
- **Higher performance:** Improve the result by many modifications for a **pure single model**(without backbone ensemble).
- **Faster training speed:** Use `gluon.data.vision.transforms` for data augmentation which is faster than original code.
- **Soft label:** Define our own cutom dataset and treat `'maybe'(m) label` as `'soft label'` when training which can boost the result.
- **Muti-scale train & test:** Use more scale augmentations when training and testing, especially for TTA(Test time augmentation). 
- **mAP defination:** Define mAP by ourselves according the competetion illustrate.
- **Random erasing:** Gluon version code defined by ourselvers.
- **Heatmap strategy:** Find the circumscribed square of the largest connected block, framing the entire heat map area for classification finetuning.


## Training in a few lines

1. Download [data](https://tianchi.aliyun.com/competition/information.htm?spm=5176.100067.5678.2.505c3a26Oet3cf&raceId=231649) and extract it into `data1/`(season1) and `data2/`(season2).
2. `python2 prepare_data.py` Prepare dataset for trainset, valset and testset.
3. `bash benchmark.sh`
  - `num_gpus`，set to 1 for single GPU training
