# [IJCAI 2020]  TextFuseNet: Scene Text Detection with Richer Fused Features 
This software implements TextFuseNet: Scene Text Detection with Richer Fused Features in PyTorch. For more details, please refer to our paper https://www.ijcai.org/Proceedings/2020/72.

## Abstract
Arbitrary shape text detection in natural scenes is an extremely challenging task. Unlike existing text detection approaches that only perceive texts based on limited feature representations, we propose a novel framework, namely TextFuseNet, to exploit the use of richer features fused for text detection. More specifically, we propose to perceive texts from three levels of feature representations, i.e., character-, word- and global-level, and then introduce a novel text representation fusion technique to help achieve robust arbitrary text detection. The multi-level feature representation can adequately describe texts by dissecting them into individual characters while still maintaining their general semantics. TextFuseNet then collects and merges the texts’ features from different levels using a multi-path fusion architecture which can effectively align and fuse different representations. In practice, our proposed TextFuseNet can learn a more adequate description of arbitrary shapes texts, suppressing false positives and producing more accurate detection results. Our proposed framework can also be trained with weak supervision for those datasets that lack character-level annotations. Experiments on several datasets show that the proposed TextFuseNet achieves state-of-the-art performance. Specifically, we achieve an F-measure of 94.3% on ICDAR2013, 92.1% on ICDAR2015, 87.1% on Total-Text and 86.6% on CTW-1500, respectively.

![image](https://github.com/ying09/TextFuseNet.pytorch/blob/master/TextFuseNet.jpg)

# Installation
This implementation is based on [Detectron2](https://github.com/facebookresearch/detectron2), the installation can refer to [step-by-step installation.txt](https://github.com/ying09/TextFuseNet/blob/master/step-by-step%20installation.txt). For more details about the environment of conda, please refer to [requirements.txt](https://github.com/ying09/TextFuseNet/blob/master/requirements.txt).

# Docker
There is also a Dockerfile for testing purposes available. See [docker](docker).

# Run demo
A demo program can be found in demo. Before running the demo, download our pretrained models from [Baidu Netdisk](https://pan.baidu.com/s/1wSjZPRh3SL1rpNMtZSHodQ) (Extraction code:8op1) or [Google Driver](https://drive.google.com/drive/folders/18Ll-3bAmi4CR2eGTuM-j6fkMrSAaBV4Z?usp=sharing). Set the path of files (include model, testing images, configs, output etc.) in demo/***_detection.py.  Then launch demo by:
    
    python demo/icdar2013_detection.py
    
the prediction outcome will come as well.
[more info](https://github.com/ying09/TextFuseNet/issues/57)


# Evaluation
Our detection code will save text contours to a txt file for each image. For calculating F-measure, Recall, and Precision, please refer to the following links:  
[ICDAR2013](https://rrc.cvc.uab.es/?ch=2)  
[ICDAR2015](https://rrc.cvc.uab.es/?ch=4)  
[Total-Text](https://github.com/cs-chan/Total-Text-Dataset)  
[CTW-1500](https://github.com/Yuliang-Liu/Curve-Text-Detector)

# Train a new model
Before training，please register your datasets in detectron2/data/datasets/builtin.py. Set training implementation details in configs/ocr/***.yaml.  To train a model with 4 gpus，please run:

    python tools/train_net.py --num-gpus 4 --config-file configs/ocr/icdar2013_101_FPN.yaml

# Annotation example
The annotation example can be found in [annotation_example](https://github.com/ying09/TextFuseNet/tree/master/annotation_example).
For word-level labels and character-level labels, please see corresponding details of weakly supervised learning method in our paper. 
For semantic segmentation labels, we generate it according to the masks of text instances during training, and for more details, please see corresponding code in [seg_head.py](https://github.com/ying09/TextFuseNet/blob/master/detectron2/modeling/roi_heads/seg_head.py).

# Results
Example results of TextFuseNet on different datasets.

![image](https://github.com/ying09/TextFuseNet/blob/master/example_results.png)


# Evaluation
Evaluation of TextFuseNet on different datasets with ResNet-101 backbone:
|Datasets|Model|Recall|Precision|F-measure|
|:------:|:------:|:------:|:------:|:------:|
|tt|Paper|85.3|89.0|87.1|
|tt|This implementation|__85.8__|__89.2__|__87.5__|
|ctw|Paper|85.4|87.8|86.6|
|ctw|This implementation|85.1|__89.7__|__87.4__|
|ic13|Paper|92.3|96.5|94.3|
|ic13|This implementation|92.1|__97.2__|__94.6__|
|ic15|Paper|89.7|94.7|92.1|
|ic15|This implementation|__90.6__|94.0|__92.2__|
|ic19-ArT|This implementation|72.8|85.4|78.6|

# Citation
    @inproceedings{ijcai2020-72,  
        title={TextFuseNet: Scene Text Detection with Richer Fused Features},  
        author={Ye, Jian and Chen, Zhe and Liu, Juhua and Du, Bo},   
        booktitle={Proceedings of the Twenty-Ninth International Joint Conference on Artificial Intelligence, {IJCAI-20}},     
        publisher={International Joint Conferences on Artificial Intelligence Organization},     
        pages={516--522},     
        year={2020}     
    }

# Acknowledgements
The authors would like to thank the developers of PyTorch and Detectron2. See [LICENSE](https://github.com/ying09/TextFuseNet/blob/master/LICENSE) for additional details.  
Please let me know if you encounter any issues.
