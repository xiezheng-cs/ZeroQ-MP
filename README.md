# ZeroQ: A Novel Zero Shot Quantization Framework
![Block](imgs/resnet18_sensitivity.png)

| ZeroQ                  | ResNet18-W6A6 | ResNet50-W6A6 | ResNet50-W4A8 | MobileNetV2-W6a6 | MobileNetV2-W4a8 | ShuffleNet-W6A6 | ShuffleNet-W4A8 |
|------------------------|---------------|---------------|---------------|------------------|------------------|-----------------|-----------------|
| size(MB)               | 8.35          | 18.27         | 12.17         | 2.50             | 1.67             | 1.11            | 0.74            |
| acc(Paper)             | 71.30         | 77.43         | 75.80         | 72.85            | 69.44            | 62.90           | 58.96           |
| acc(train)             | 70.81         | 66.72         | 73.37         | 72.14            | 50.42            | 46.79           | 52.82           |
| acc(distill)           | 69.86         | 70.97         | 72.72         | 72.14            | 53.54            | 36.42           | 50.53           |
| acc(train)per0.999     | 69.04         | 74.24         | 71.55         | 71.86            | 59.81            | 61.33           | 51.34           |
| acc(train)per0.9999    | 71.12         | 76.78         | 73.63         | 72.05            | 60.26            | 62.19           | 52.26           |
| acc(train)per0.99999   | 71.16         | 76.97         | 73.62         | 72.18            | 59.12            | 60.64           | 52.72           |
| acc(distill)per0.99999 | 71.33         | 76.27         | 72.97         | 72.20            | 53.48            | 59.58           | 50.97           |
| acc(distill)per0.9999  | 70.96         | 76.27         | 72.48         | 72.20            | 53.19            | 61.59           | 50.59           |

## Introduction

This repository contains the PyTorch implementation for the **CVPR 2020** paper [*ZeroQ: A Novel Zero-Shot Quantization Framework*](https://arxiv.org/abs/2001.00281). Below are instructions for reproducing classification results. Please see [detection readme](https://github.com/amirgholami/ZeroQ/tree/master/detection) for instructions to reproduce object detection results.

You can find a short video explanation of ZeroQ [here](https://news.developer.nvidia.com/nvidia-partners-present-ai-research-at-cvpr-2020/).

## TLDR;

```bash
# Code is based on PyTorch 1.2 (Cuda10). Other dependancies could be installed as follows: 
cd classification
pip install -r requirements.txt --user
# Set a symbolic link to ImageNet validation data (used only to evaluate model) 
mkdir data
ln -s /path/to/imagenet/ data/
```

The folder structures should be the same as following
```
zeroq
├── utils
├── data
│   ├── imagenet
│   │   ├── val
```
Afterwards you can test Zero Shot quantization with W8A8 by running:

```bash
bash run.sh
```

Below are the results that you should get for 8-bit quantization (**W8A8** refers to the quantizing model to 8-bit weights and 8-bit activations).


| Models                                          | Single Precision Top-1 | W8A8 Top-1 | W8A8 Top-1 | Single Precision Top-1 |   
| ----------------------------------------------- | :--------------------: | :--------: | :--------: | :--------------------: |
| [ResNet18](https://arxiv.org/abs/1512.03385)    |          71.47         |   71.43    |  71.614    |     71.97              |
| [ResNet50](https://arxiv.org/abs/1512.03385)    |          77.72         |   77.67    |  77.542    |     77.72              |
| [InceptionV3](https://arxiv.org/abs/1512.00567) |          78.88         |   78.72    |            |                        |
| [MobileNetV2](https://arxiv.org/abs/1801.04381) |          73.03         |   72.91    |            |     73.03              |
| [ShuffleNet](https://arxiv.org/abs/1707.01083)  |          65.07         |   64.94    |  64.478    |     65.07              |
| [SqueezeNext](https://arxiv.org/abs/1803.10615) |          69.38         |   69.17    |  68.768    |     69.38              |

## Evaluate

- You can test a single model using the following command:

```bash
export CUDA_VISIBLE_DEVICES=0
python uniform_test.py [--dataset] [--model] [--batch_size] [--test_batch_size]

optional arguments:
--dataset                   type of dataset (default: imagenet)
--model                     model to be quantized (default: resnet18)
--batch-size                batch size of distilled data (default: 64)
--test-batch-size           batch size of test data (default: 512)
```




## Citation
ZeroQ has been developed as part of the following paper. We appreciate it if you would please cite the following paper if you found the implementation useful for your work:

```text
@inproceedings{cai2020zeroq,
  title={Zeroq: A novel zero shot quantization framework},
  author={Cai, Yaohui and Yao, Zhewei and Dong, Zhen and Gholami, Amir and Mahoney, Michael W and Keutzer, Kurt},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={13169--13178},
  year={2020}
}
```
