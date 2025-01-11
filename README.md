
# Research on Synthetic Speech Detection Technology Based on Deep Neural Network
This is my undergraduate project. In this paper, based on different types of forgery, we construct a time-domain forgery speech detection model based on attention mechanism and a generalized forgery speech detection model based on time-frequency domain features, which improves the accuracy, generalization and universality of the models.
# 实验主要内容
**（1）为了提高模型在逻辑访问上的检测准确率，提出了基于注意力机制的时域伪造语音检测模型。**
在原有基于并行卷积的时域伪造语音检测模型的基础上，引入注意力机制，实现了模型对不同通道或空间区域特征学习的自动调整。为了增强时频域中的特征表示，将挤压-激励块（Squeeze-and-Excitation，SE）和空间组别增强块（Spatial Group-wise Enhance，SGE）并嵌入并行卷积中，分别提出了两种改进模型。在ASVspoof2019测试集上对提出的两种检测模型进行了性能测试，并与原模型进行了对比。实验结果表明，提出的嵌入SE和SGE的改进检测模型均具有良好的检测性能，且均优于原模型（其等错误率较原模型相对降低4.95%和4.05%）。
**（2）为使得模型同时检测逻辑访问和物理访问的伪造语音，提出了基于时频域特征的伪造语音通用检测模型。**
为学习更深层的特征，该模型采用残差网络作为核心网络，并嵌入调整的挤压-激励块，产生多个尺度特征并建立通道相关性，实现了更为高效的检测。此外，分别对线性频率倒谱系数、常数Q变换、常数Q倒谱系数声学特征进行了提取，并作为模型输入在ASVspoof2019测试集上进行了综合测试，并与6种逻辑访问专用检测方法和6种物理访问专用检测方法进行了对比。实验结果表明，基于常数Q变换的模型具有最佳的通用性能；较之已有工作，基于常数Q变换的伪造语音通用检测模型不仅可实现“一次训练，通用检测”的目标，且可达到较好的检测性能（平均检测正确率可以达到93.85%）。
**（3）设计并实现了伪造语音检测系统，该系统分为交互层、检测层和数据层，**
其中交互层采用Vue框架，实现了选择检测模型、上传检测样本、播放检测样本、查看检测报告的功能；检测层集成了两种检测模型，实现了语音采样、特征提取、获取检测结果的功能；数据层旨在存储和读取数据，确保存储的语音信号信息被高效管理。通过广泛的功能测试表明该系统达到了预期的设计需求，具有良好的可操作性和可靠性。

## 模型运行
### Data Preparation 
```
ASVspoof15&19_LA_Data_Preparation.py
```
### Training 
```
train.py
```
### Testing
```
test.py
```
## fake speech detection system系统运行

## Project setup
```
npm install
```

### Compiles and hot-reloads for development
```
npm run serve
```
## 系统展示
https://duantanghui.github.io/Synthetic-Speech-Detection/#/home
### 考虑到后端无法放到云服务器，这里的最终报告是写死的。
