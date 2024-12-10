# 基于时频域特征的伪造语音通用检测模型

# 实验主要内容

- **CQT/LFCC/CQCC+SE+Res-TSSDNet ASVspoof2019LA和PA的混合集上训练和测试**
- **CQT+SE+Res-TSSDNet ASVspoof2019混合集训练但在LA、PA测试集上分别测试**
- **CQT+SE+Res-TSSDNet ASVspoof2019LA、PA数据集上分别训练和测试**

## 步骤
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
