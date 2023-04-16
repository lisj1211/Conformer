## Introduction

基于Pytorch的Conformer语音识别模型实现, 相比于[DeepSpeech2](https://github.com/lisj1211/DeepSpeech2), 增加了长语音和流式识别. 运行环境推荐linux环境, 部分模型只支持linux系统. 

## Requirements

* python 3.7
* torch = 1.12.1
* numpy
* tqdm
* yaml
* cn2an
* termcolor
* Levenshtein
* typeguard
* paddlespeech_ctcdecoders(仅限linux平台)
* ffmpeg
* resampy
* scipy
* pydub
* zhconv
* torchaudio
* pillow

## DataSet

数据集为aishell,178小时的中文语音数据集.
语音数据[aishell](https://openslr.magicdatatech.com/resources/33/data_aishell.tgz)
, 噪声数据[noise](http://www.openslr.org/resources/28/rirs_noises.zip), 如果不需要数据增强操作可以不用噪声数据.
下载完成后将两个压缩文件放至`data`文件夹下`

## Train

* 数据预处理, 分别对语音和噪声数据进行解压缩操作, 之后运行`create_data.py`进行数据预处理操作.
  模型配置文件为`configs/conformer.yml`, 数据扩充配置文件为`config/augmentation.json`

```
    cd ./data_preprocess
    python aishell.py
    python noise.py
    cd ..
    python create_data.py
```

* 训练模型

```
    python train.py
```

* 导出模型

```
    python export_model.py
```

* 模型预测

```
    python infer.py
```

## Results

|             | Val_cer | Test_cer |
|:------------|:--------|:---------|
| DeepSpeech2 | 0.08212 | 0.0936   |
| Conformer   | 0.05752 | 0.0421   |

cer表示字错率, 即预测文本与真实文本之间的编辑距离

## Analysis

测试结果对于长语音识别效果不好, 可能是VAD语音活动检测工具的问题. 对于BeamSearch参数的调优见`tune.py`. 测试时先运行`export_model.py`导出静态模型, 之后运行`infer.py`进行不同模式的语音识别.
## Reference

[1] [MASR](https://github.com/yeyupiaoling/MASR)
