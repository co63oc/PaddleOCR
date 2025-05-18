# 通用数学公式识别模型-TrOCR-Formula-Rec

## 1. 简介

原始项目：
> [https://github.com/SWHL/TrOCR-Formula-Rec](https://github.com/SWHL/TrOCR-Formula-Rec)


`TrOCR-Formula-Rec`使用[`UniMERNet通用公式识别数据集`](https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main)进行训练，在对应测试集上的精度如下：

| 模型        | SPE-<br/>BLEU↑ | SPE-<br/>EditDis↓ | CPE-<br/>BLEU↑  |CPE-<br/>EditDis↓ | SCE-<br/>BLEU↑ | SCE-<br/>EditDis↓ | HWE-<br/>BLEU↑ | HWE-<br/>EditDis↓ | 下载链接 |
|-----------|------------|-------------------------------------------------------|:--------------:|:-----------------:|:----------:|:----------------:|:---------:|:-----------------:|:--------------:|:-----------------:|-------|
| UniMERNet |     0.9187     |      0.0584       |  0.9252    |      0.0596      | 0.6068 |     0.2297        |   0.9157|     0.0546           |[训练模型](https://huggingface.co/co63oc/trocr-paddle)|

其中，SPE表示简单公式，CPE表示复杂公式，SCE表示扫描捕捉公式，HWE表示手写公式。每种类型的公式示例图如下：
![unimernet_dataset](https://github.com/user-attachments/assets/fb801a36-5614-4031-8585-700bd9f8fb2e)

## 2. 环境配置
请先参考[《运行环境准备》](../../ppocr/environment.md)配置PaddleOCR运行环境，参考[《项目克隆》](../../ppocr/blog/clone.md)克隆项目代码。

此外，需要安装额外的依赖：
```shell
# PaddleNLP 和 PaddleMIX 都有修改，如果修改没有合入开发分支，需要clone代码安装，分支名称为 trocr
git clone https://github.com/co63oc/PaddleNLP/
cd PaddleNLP
git checkout trocr
pip install -e .

git clone https://github.com/co63oc/PaddleMIX/
cd PaddleMIX
git checkout trocr
pip install -e .
cd ppdiffusers
pip install -e .

# 原仓库中测试用到依赖
pip install sentencepiece \
datasets \
jiwer \
tabulate \
Pillow \
tqdm \
protobuf \
evaluate \
albumentations \
bleu
```

## 3. 模型训练、评估、预测

### 3.1 准备数据集

从 [Hugging Face](https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main) 上下载 UniMER-1M.zip 和 UniMER-Test.zip。
从 [好未来平台](https://ai.100tal.com/dataset) 下载 HME100K 数据集。

```shell
# 解压 UniMER-1M 、 UniMER-Test
cd tests/trocr/
mkdir dataset
unzip -d dataset/ path/UniMER-1M.zip
unzip -d dataset/ path/UniMER-Test.zip
unzip -d dataset/UniMER-1M/HME100K/ path/HME100K/train.zip
unzip -d dataset/UniMER-1M/HME100K/ path/HME100K/test.zip
```

### 3.2 下载预训练模型

```shell
# 下载预训练模型
cd tests/trocr/
# token解析模型，不用转换权重，从huggingface下载到工作目录
huggingface-cli download --local-dir trocr-small-stage1 microsoft/trocr-small-stage1
# 使用 https://huggingface.co/SWHL/TrOCR-Formula-Rec/tree/main/Exp8 转换后的权重，转换脚本 tests/trocr/convert_weight.py
huggingface-cli download --local-dir trocr-paddle co63oc/trocr-paddle
```


### 3.3 模型训练

#### 启动训练


具体地，在完成数据准备后，便可以启动训练，训练命令如下：
```shell
#单卡训练 (默认训练方式)
python3 tools/train.py -c configs/rec/UniMERNet.yaml \
   -o Global.pretrained_model=./pretrain_models/texify.pdparams
#多卡训练，通过--gpus参数指定卡号
python3 -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/UniMERNet.yaml \
        -o Global.pretrained_model=./pretrain_models/texify.pdparams
```

**注意：**

- 默认每训练 1个epoch（37880 次iteration）进行1次评估，若您更改训练的batch_size，或更换数据集，请在训练时作出如下修改
```
python3  -m paddle.distributed.launch --gpus '0,1,2,3' --ips=127.0.0.1   tools/train.py -c configs/rec/UniMERNet.yaml \
  -o Global.eval_batch_step=[0,{length_of_dataset//batch_size//4}] \
   Global.pretrained_model=./pretrain_models/texify.pdparams
```

### 3.4 评估

使用如下命令进行评估：

```shell
cd tests/trocr/
python ./test.py
```

### 3.5 预测

使用如下命令进行单张图片预测：
```shell
# 修改文件中不同图片路径测试。
cd tests/trocr/
python ./predict.py
```

## 4. FAQ

1. UniMERNet 数据集来自于[UniMERNet源repo](https://github.com/opendatalab/UniMERNet) 。
