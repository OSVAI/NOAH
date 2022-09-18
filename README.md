# NOAH: Improving Deep Neural Networks For Image Classification Via Non-Global Attentive Head
By Chao Li, Aojun Zhou and Anbang Yao.

This repository is an official PyTorch implementation of ["NOAH: Improving Deep Neural Networks For Image Classification Via Non-Global Attentive Head"](https://arxiv.org/abs/xxxx.xxxxx). 

A modern deep neural network (DNN) for image classification typically consists of two parts: a backbone for feature extraction, and a head for feature encoding and output predication. We notice that the head structures of prevailing DNNs in general share a similar processing pipeline, exploiting global feature dependencies while disregarding local ones. Instead, in this project, we present **N**on-gl**O**bal **A**ttentive **H**ead (NOAH, for short), a simple and universal head structure, to improve the learning capacity of DNNs. NOAH relies on a novel form of attention dubbed **Pairwise Object Category Attention**, which models dense local-to-global feature dependencies in a group-wise manner via a concise association of feature split, interaction and aggregation operations. As a drop-in design, NOAH can replace existing heads of many DNNs, and meanwhile, maintains almost the same model size and similar model effiency. We validate the efficacy of NOAH on the large-scale ImageNet dataset with various DNN architectures that span convolutional neural networks, vision transformers and multi-layer perceptrons when training from scratch. Without bells and whistles, experiments show that: (a) NOAH can significantly boost the performance of lightweight DNNs, e.g., bringing **3.14%|5.30%|1.90%** top-1 accuracy improvement for MobileNetV2 (0.5X)|Deit-Tiny (0.5X)|gMLP-Tiny (0.5X); (b) NOAH can generalize well on relatively large DNNs, e.g., bringing **1.02%|0.78%|0.91%** top-1 accuracy improvement for ResNet50|Deit-Small|MLP-Mixer-Small; (c) NOAH can still bring acceptable performance gains to large DNNs (having over 50 million parameters), e.g., **0.41%|0.37%|0.35%** top-1 accuracy improvement for ResNet152|Deit-Base|MLP-Mixer-Base. Besides, NOAH also retains its effectiveness in the aggressive training regime (e.g., a ResNet50 attains **79.32%** top-1 accuracy on ImageNet) and other image classification tasks. Code and models will be released soon.

<p align="center"><img src="fig/noah_architecture.png" width="800" /></p>
The macro-structure of DNNs with a Non-glObal Attentive Head (NOAH). Unlike popular heads using the global feature encoding, NOAH relies on Pairwise Object Category Attentions (POCAs) learnt at local to global scales via a neat association of feature split (two levels), interaction and aggregation operations, taking the feature maps from the last layer of any backbone as the input.
