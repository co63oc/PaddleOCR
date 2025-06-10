# TrOCR-Formula-Rec

## 1. Introduction

Original Project：
> [https://github.com/SWHL/TrOCR-Formula-Rec](https://github.com/SWHL/TrOCR-Formula-Rec)

`TrOCR-Formula-Rec` was trained using the [`UniMERNet Universal Formula Recognition Dataset`](https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main), and its accuracy on the corresponding test set is as follows:

| Model        | SPE-<br/>BLEU↑ | SPE-<br/>EditDis↓ | CPE-<br/>BLEU↑  |CPE-<br/>EditDis↓ | SCE-<br/>BLEU↑ | SCE-<br/>EditDis↓ | HWE-<br/>BLEU↑ | HWE-<br/>EditDis↓ | Download link |
|-----------|------------|-------------------------------------------------------|:--------------:|:-----------------:|:----------:|:----------------:|:---------:|:-----------------:|:--------------:|:-----------------:|-------|
| TrOCR-Formula-Rec |     0.886     |      0.069       |  0.822    |      0.108      | 0.634 |     0.216        |   0.897|     0.067           |[Trained model](https://huggingface.co/co63oc/trocr-paddle)|

SPE represents simple formulas, CPE represents complex formulas, SCE represents scanned captured formulas, and HWE represents handwritten formulas. Example images of each type of formula are shown below:

![unimernet_dataset](https://github.com/user-attachments/assets/fb801a36-5614-4031-8585-700bd9f8fb2e)

## 2. Environment
Please refer to ["Environment Preparation"](../../ppocr/environment.en.md) to configure the PaddleOCR environment, and refer to ["Project Clone"](../../ppocr/blog/clone.en.md) to clone the project code.

Furthermore, additional dependencies need to be installed:

```shell
# Both PaddleNLP and PaddleMIX have undergone modifications. If the modifications have not been merged into the development branch, you need to clone the code for installation, and the branch name is trocr
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

# Dependencies used for testing in the original repository
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

## 3. Model Training / Evaluation / Prediction

Please refer to [Text Recognition Tutorial](../../ppocr/model_train/recognition.en.md). PaddleOCR modularizes the code, and training different recognition models only requires **changing the configuration file**.

Dataset Preparation:

Download UniMER-1M.zip and UniMER-Test.zip from [Hugging Face]((https://huggingface.co/datasets/wanderkid/UniMER_Dataset/tree/main)). Download the HME100K dataset from the [TAL AI Platform](https://ai.100tal.com/dataset). After that, use the following command to create a dataset directory and convert the dataset.

```shell
# unzip UniMER-1M、UniMER-Test
cd tests/trocr/
mkdir dataset
unzip -d dataset/ path/UniMER-1M.zip
unzip -d dataset/ path/UniMER-Test.zip
unzip -d dataset/UniMER-1M/HME100K/ path/HME100K/train.zip
unzip -d dataset/UniMER-1M/HME100K/ path/HME100K/test.zip
```

Download the Pre-trained Model:

```shell
cd tests/trocr/
# Download the token parsing model directly from HuggingFace to the working directory without converting the weights.
huggingface-cli download --local-dir trocr-small-stage1 microsoft/trocr-small-stage1
# Use the converted weights from https://huggingface.co/SWHL/TrOCR-Formula-Rec/tree/main/Exp8. Conversion script is tests/trocr/convert_weight.py
huggingface-cli download --local-dir trocr-paddle co63oc/trocr-paddle
```

### 3.3 Training

#### Training

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

### 3.4 Evaluation

Evaluation:

```shell
cd tests/trocr/
python ./test.py
```

### 3.5 Prediction

```shell
# Change the image path in predict.py to your own path
cd tests/trocr/
python ./predict.py
```

## 4. FAQ
