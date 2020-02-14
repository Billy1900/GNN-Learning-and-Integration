# GNN-Learning-and-Integration

- [x] [How to read paper](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/How%20to%20Read%20a%20Paper.pdf)

## GNN 入门
- [从图论->GNN，很初级,GNN讲的很一般,基本的图论偏多](https://www.bilibili.com/video/av62661713/?spm_id_from=333.788.videocard.3)
- [Jure Leskovec, Computer Science Department, Stanford University](https://www.bilibili.com/video/av51673220/?spm_id_from=333.788.videocard.1)

- [x] [GNN Introduction 中文版](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/GNN_Review1.1.pdf)
- [x] [GNN 综述(比较全)](https://zhuanlan.zhihu.com/p/76001080)
- [x] [浅显易懂](https://zhuanlan.zhihu.com/p/38612863)
- [x] [进阶--各个模型的总结](https://zhuanlan.zhihu.com/p/65539782)
- [x] [基本概念扫盲](https://zhuanlan.zhihu.com/p/54505069)
- [x] [英文版 视频图片助于理解](http://tkipf.github.io/graph-convolutional-networks/)
- [x] [GNN三代演进](http://xtf615.com/2019/02/24/gcn/)
- [x] [如何理解 Graph Convolutional Network（GCN）?:重点是卷积以及傅里叶的理解](https://www.zhihu.com/question/54504471/answer/332657604)
- [x] [GNN Intro slides](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/gnn%20Intro.pdf)

## 进阶：经典论文
- [x] [Graph Neural Networks-A Review of Methods and Applications.pdf](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Graph%20Neural%20Networks-A%20Review%20of%20Methods%20and%20Applications.pdf)
  - [x] [《The graph neural network model》](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/The%20graph%20neural%20network%20model.pdf)
  - [x] [论文《The Graph Neural Network Model》中GNN模型及实现细节](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/%E8%AE%BA%E6%96%87%E3%80%8AThe%20Graph%20Neural%20Network%20Model%E3%80%8B%E4%B8%ADGNN%E6%A8%A1%E5%9E%8B%E5%8F%8A%E5%AE%9E%E7%8E%B0%E7%BB%86%E8%8A%82.pdf)
  - [ ] [Diffusion-Convolutional Neural Networks.pdf](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Diffusion-Convolutional%20Neural%20Networks.pdf)
  - [ ] [Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks.pdf](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Improved%20Semantic%20Representations%20From%20Tree-Structured%20Long%20Short-Term%20Memory%20Networks.pdf)
  - [ ] [semi_supervised_classification_with_graph_convolutional_networks.pdf](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/semi_supervised_classification_with_graph_convolutional_networks.pdf)
  - [ ] [Variational Graph Auto-Encoders.pdf](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Variational%20Graph%20Auto-Encoders.pdf)
- [x] [Must read paper in GNN](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Must_read_paper_GNN.md)

## Pytorch框架
- PyTorch Geometric
  - [Installation Documentation](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)
  - [source code (github repo)](https://github.com/rusty1s/pytorch_geometric)
  - [数据集 dataset](https://linqs.soe.ucsc.edu/data)
  - [PyG教程](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/Hands-on-Graph-Neural-Networks-with-PyTorch-PyTorch-Geometric1.pdf)
- [x] [PyG和Pytorch实现GNN模型code](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/PyG%E5%92%8CPytorch%E5%AE%9E%E7%8E%B0GNN%E6%A8%A1%E5%9E%8B.zip)

## Machine Learning
- [x] [Machine Learning Methods](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/new-in-ml-2019.pdf)
专家提供，有很大参考意义
- [x] [机器学习三大框架资源](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E4%B8%89%E5%A4%A7%E6%A1%86%E6%9E%B6%E5%AD%A6%E4%B9%A0.md)

-------------------------------------------------------------------------------------------------------------------------------------

## 2020-02-13
此时已经基本明白了GNN的数学原理，要做的是梳理GNN的发展流程，以下三篇博客是十分好的材料
- [从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (一)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_1.html)
- [从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (二)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_2.html)
- [从图(Graph)到图卷积(Graph Convolution)：漫谈图神经网络模型 (三)](https://www.cnblogs.com/SivilTaram/p/graph_neural_network_3.html)
- [x] [GNN Introduction 中文版](https://github.com/Billy1900/GNN-Learning-and-Integration/blob/master/GNN_Review1.1.pdf)

<pre>
卷积
	对卷积的意义的理解：
	1. 从“积”的过程可以看到，我们得到的叠加值，是个全局的概念。以信号分析为例，卷积的结果是不仅跟当前时刻输入信号的响应值有关，也跟过去所有时刻输入信号的响应都有关系，考虑了对过去的所有输入的效果的累积。在图像处理的中，卷积处理的结果，其实就是把每个像素周边的，甚至是整个图像的像素都考虑进来，对当前像素进行某种加权处理。所以说，“积”是全局概念，或者说是一种“混合”，把两个函数在时间或者空间上进行混合。
	2. 那为什么要进行“卷”？直接相乘不好吗？我的理解，进行“卷”（翻转）的目的其实是施加一种约束，它指定了在“积”的时候以什么为参照。在信号分析的场景，它指定了在哪个特定时间点的前后进行“积”，在空间分析的场景，它指定了在哪个位置的周边进行累积处理
</pre>
