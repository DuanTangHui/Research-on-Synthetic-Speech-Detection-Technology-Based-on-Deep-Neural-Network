# 基于注意力机制的时域伪造语音检测模型

# 实验结果

- **SE+Inc-TSSDNet ASVspoof2019 LA eval EER: 4.738%;**
- **SGE+Inc-TSSDNet ASVspoof2019 LA eval EER: 4.783%.**


## 数据集
  
1. ASVspoof 2019 train set is used for training;
2. ASVspoof 2019 dev set is used for model selection;
3. ASVspoof 2019 eval set is used for testing;

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
