# GNN-Learning-and-Integration
First, for beginners, I really recommend they should start from a college course, like [CS224W: Machine Learning with Graphs, stanford Fall 2019](http://web.stanford.edu/class/cs224w/) which could help you get a good understanding of networks and how we discover information out of it.
## 1. GNN Intuitive Learning
- [CNN explainer](https://github.com/poloclub/cnn-explainer)
- [GNN model explainer](https://github.com/RexYing/gnn-model-explainer)
- [Fundamental graph theory](https://www.bilibili.com/video/av62661713/?spm_id_from=333.788.videocard.3)
- [what is Convolution, graph Laplacian?](https://zhuanlan.zhihu.com/p/54505069)
- [Graph Neural Network by kipf](http://tkipf.github.io/graph-convolutional-networks/)
- [GCN Introduction](https://zhuanlan.zhihu.com/p/120311352)
- [GCN 为什么是低通滤波器](https://zhuanlan.zhihu.com/p/142640571)
- 从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型
  - [从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (一)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_1.html)
  - [从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (二)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_2.html)
  - [从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (三)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_3.html)

## 2. GNN Mathematical Theory Learning
- [GNN Conclusions](https://zhuanlan.zhihu.com/p/76001080)
- [GNN Review report](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/GNN_Review1.1.pdf)
- [Graph model: graph embedding and graph convolutional network](https://zhuanlan.zhihu.com/p/65539782)
- [Mathematical foundation of GNN](http://xtf615.com/2019/02/24/gcn/)
- [Dive into Convolution deeply: Mathematical derivation](https://www.zhihu.com/question/54504471/answer/332657604)

## 3. Academic Paper
### 3.1 Survey Paper
- [Graph Neural Networks-A Review of Methods and Applications.pdf](https://arxiv.org/abs/1812.08434)
- [Deep Learning on Graphs: A Survey](https://arxiv.org/pdf/1812.04202.pdf)
- [A Comprehensive Survey on Graph Neural Networks](https://arxiv.org/abs/1901.00596)
- [Machine Learning on Graphs: A Model and Comprehensive Taxonomy](https://arxiv.org/pdf/2005.03675.pdf)
- [A Practical Guide to Graph Neural Networks](https://arxiv.org/abs/2010.05234)
- [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)
### 3.2 Some Important Papers on GNN
- The graph neural network model
  - [The graph neural network model](http://persagen.com/files/misc/scarselli2009graph.pdf)
  - [The Graph Neural Network Model explanation](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/%E8%AE%BA%E6%96%87%E3%80%8AThe%20Graph%20Neural%20Network%20Model%E3%80%8B%E4%B8%ADGNN%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82.pdf)
- [Diffusion-Convolutional Neural Networks](https://papers.nips.cc/paper/6212-diffusion-convolutional-neural-networks.pdf)
- [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks](https://arxiv.org/abs/1503.00075)
- Semi Supervised Classification With Graph Convolutional Networks (GCN)
  - [Semi Supervised Classification With Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)
  - [GCN code explanation](https://blog.csdn.net/yyl424525/article/details/100634211)
- [Variational Graph Auto-Encoders.pdf](https://arxiv.org/abs/1611.07308)

## 4. Survey
- [Must read paper in GNN](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Must_read_paper_GNN.md)
- [Awesome graph neural networks paper list](https://github.com/nnzhan/Awesome-Graph-Neural-Networks)

## 5. Tools
### 5.1 Three Tools
  - Tensorflow
    - [How to Install Tensorflow 2.1.0 in windows10?： CUDA 10.1, CUDnn 7.6](https://blog.csdn.net/weixin_44170512/article/details/103990592)
    - [Medium--Migrating tensorflow 1.x to tensorflow 2.x.](https://medium.com/tensorflow/upgrading-your-code-to-tensorflow-2-0-f72c3a4d83b5)
    - [TensorFlow Tutorial and Examples for Beginners (support TF v1 & v2)](https://github.com/aymericdamien/TensorFlow-Examples)
  - Keras
  - Pytorch
    - [Awesome-pytorch-list](https://github.com/bharathgs/Awesome-pytorch-list): A comprehensive list of pytorch related content on github,such as different models,implementations,helper libraries,tutorials etc
    - [How to solve problem: No module named torch_sparse](https://zhuanlan.zhihu.com/p/163180187)
### 5.2 Dataset
  - [Dataset library](https://linqs.soe.ucsc.edu/data)
  - [Cora Introduction](https://blog.csdn.net/yeziand01/article/details/93374216)
  - [Stanford Large Network Dataset Collection](https://snap.stanford.edu/data/)
  - [planetoid](https://github.com/kimiyoung/planetoid)
### 5.3 Library to build GNN easily
- [Deep Graph Library (DGL)](https://github.com/dmlc/dgl)
- [Graph Neural Networks with Keras and Tensorflow 2.](https://github.com/danielegrattarola/spektral): Spektral is a Python library for graph deep learning, based on the Keras API and TensorFlow 2. The main goal of this project is to provide a simple but flexible framework for creating graph neural networks (GNNs).
- [CogDL: An Extensive Research Toolkit for Graphs](https://github.com/THUDM/cogdl/)
- [Graph Convolutional Neural Networks (GCNN) models](https://github.com/google/gcnn-survey-paper): This repository contains a tensorflow implementation of GCNN models for node classification, link predicition and joint node classification and link prediction to supplement the [survey paper by Chami et al.](https://arxiv.org/pdf/2005.03675.pdf)
- Benchmarking Graph Neural Networks [[paper]](https://arxiv.org/pdf/2003.00982v3.pdf) [[code]](https://github.com/graphdeeplearning/benchmarking-gnns)
- Stella Graph [[documentation]](https://stellargraph.readthedocs.io/en/stable/index.html) [[github repo]](https://github.com/stellargraph/stellargraph)
- PyCaret 2.0 [[link]](https://github.com/pycaret/pycaret)
- graph_star [[link]](https://github.com/graph-star-team/graph_star)
### 5.4 To draw
- Matplotlib教程 [[Link]](https://morvanzhou.github.io/tutorials/data-manipulation/plt/)
- [How to use t-SNE efficiently](https://distill.pub/2016/misread-tsne/)
- [Scikit-plot](https://github.com/reiinakano/scikit-plot)
- [Tools-to-Design-or-Visualize-Architecture-of-Neural-Network](https://github.com/ashishpatel26/Tools-to-Design-or-Visualize-Architecture-of-Neural-Network)
- ML Visuals [[repo]](https://github.com/dair-ai/ml-visuals) [[slides]](https://docs.google.com/presentation/d/11mR1nkIR9fbHegFkcFq8z9oDQ5sjv8E3JJp1LfLGKuk/edit#slide=id.g85a0789696_743_21)
- [Science Plots](https://github.com/garrettj403/SciencePlots)
- [Visualize of loss function](https://izmailovpavel.github.io/curves_blogpost/)

## 6. Courses & Learning material
- [吴恩达机器学习系列](https://zhuanlan.zhihu.com/p/108243142)
- [AlphaTree-graphic-deep-neural-network](https://github.com/weslynn/AlphaTree-graphic-deep-neural-network): 机器学习(Machine Learning)、深度学习(Deep Learning)、对抗神经网络(GAN），图神经网络（GNN），NLP，大数据相关的发展路书(roadmap), 并附海量源码（python，pytorch）带大家消化基本知识点
- [Interpretable Machine Learning](https://github.com/christophM/interpretable-ml-book)
- [深度学习 AI圣经(Deep Learning)](https://github.com/MingchaoZhu/DeepLearning)
- [Machine-Learning-Book（机器学习宝典）](https://github.com/yuanxiaosc/Machine-Learning-Book)

- [DeepLearning-500-questions](https://github.com/scutan90/DeepLearning-500-questions)
- [Awesome Math](https://github.com/llSourcell/learn_math_fast): A curated list of awesome mathematics resources.
- [Graph Representation Learning Book](https://www.cs.mcgill.ca/~wlh/grl_book/)
- [Deep Learning on Graphs](https://cse.msu.edu/~mayao4/dlg_book/)
- [Virgilio Data Science](https://github.com/virgili0/Virgilio)
- [ML-From-Scratch](https://github.com/eriklindernoren/ML-From-Scratch)
- [C5.4 Networks From Harvard](https://courses.maths.ox.ac.uk/node/view_material/47273)

## 7. Other
- [Research Methods in Machine Learning](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/new-in-ml-2019.pdf)
- [批量梯度下降(BGD)、随机梯度下降(SGD)、小批量梯度下降(MBGD)](https://zhuanlan.zhihu.com/p/72929546)
- [损失函数可视化及其对神经网络的指导作用](https://zhuanlan.zhihu.com/p/52314278)
- [如何理解dropout](https://blog.csdn.net/stdcoutzyx/article/details/49022443)
- [Implementation and experiments of graph embedding algorithms](https://github.com/shenweichen/GraphEmbedding)
- [Inductive vs. Transductive Learning](https://towardsdatascience.com/inductive-vs-transductive-learning-e608e786f7d)


## 8. Graph Adversarial Learning
- [Awesome Graph Adversarial Learning (Updating)](https://github.com/gitgiter/Graph-Adversarial-Learning)
- [Awesome Graph Attack and Defense Papers](https://github.com/Billy1900/GCN-DP/blob/master/Awesome%20Graph%20Attack%20and%20Defense%20Papers.md)
- [DeepRobust: a repository with representative algorithms on Graph attack and defense model](https://github.com/DSE-MSU/DeepRobust/tree/master/deeprobust/graph)
- [A curated list of adversarial attacks and defenses papers on graph-structured data](https://github.com/safe-graph/graph-adversarial-learning-literature)
- [图对抗攻击 Graph Adversarial Attack--zhihu](https://zhuanlan.zhihu.com/p/88934914)
