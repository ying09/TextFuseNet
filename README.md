# TextFuseNet: Scene Text Detection with Richer Fused Features
This software implements TextFuseNet: Scene Text Detection with Richer Fused Features in pytorch. For details, please refer to our paper https://www.ijcai.org/Proceedings/2020/72.

## Abstract
Arbitrary shape text detection in natural scenes is an extremely challenging task. Unlike existing text detection approaches that only perceive texts based on limited feature representations, we propose a novel framework, namely TextFuseNet, to exploit the use of richer features fused for text detection. More specifically, we propose to perceive
texts from three levels of feature representations, i.e., character-, word- and global-level, and then introduce a novel text representation fusion technique to help achieve robust arbitrary text detection. The multi-level feature representation can adequately describe texts by dissecting them into individual characters while still maintaining their
general semantics. TextFuseNet then collects and merges the texts’ features from different levels using a multi-path fusion architecture which can effectively align and fuse different representations. In practice, our proposed TextFuseNet can learn a more adequate description of arbitrary shapes texts, suppressing false positives and producing more accurate detection results. Our proposed framework can also be trained with weak supervision for those datasets that lack character-level annotations. Experiments on several datasets show that the proposed TextFuseNet achieves state-of-the-art performance. Specifically, we achieve an F-measure of 94.3% on ICDAR2013, 92.1% on ICDAR2015, 87.1% on Total-Text and 86.6% on CTW-1500, respectively.

![image](https://github.com/ying09/TextFuseNet.pytorch/blob/master/TextFuseNet.jpg)

# Installation
Our sofeware is based on Detectron2, the installation can refer to https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md.

# Run demo

# Train a new model

# Annotation sample

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
The authors would like to thank the developers of PyTorch and Detectron2. Please let me know if you encounter any issues.
