# DLPAP
Code for the paper `Deep Learning for Patent Application: The Fusion of Network and Text Embedding`

### 项目文件说明：
- data_process：数据处理
   - class_count.py：提取patent application中的cpc分类信息，按照每一年进行统计
   - index_build.py：遍历patent application文本，建立patent-inventor、patent-assignee之间的关系作为索引，按照每一年进行统计
   - split_citations.py：对PatentsView中统计的citations信息按照每一年进行划分和统计
   - xmlfile_deal.py：对patent application的原始zip文件进行数据预处理

- sample：样本数据采集
  - get_class.py
  - get_network.py
  - get_parents.py
  - get_textual.py
  - node.ipynb：节点数据的处理，建立唯一ID的对应关系

- sample_data：文本使用的数据
  - network
  - pre-trained model：GloVe-300d pre-trained model
  - textual

- model_dlpap：文本使用的模型DLPAP
  - ActiveHNE：使用ActiveHNE提取网络特征，原文是`Xia Chen, Guoxian Yu, et al. Activehne: Active heterogeneous network embedding. In Proc. of IJCAI, 2019`
  - Contextual BiLSTM：使用Contextual BiLSTM提取文本特征
  - Corpus process+TF-IDF vector.ipynb：文本语料预处理
  - Full-connect Network.ipynb：全连接网络预测层
  - corpus_process.py

- baseline_TF-IDF
  - Baseline：TF-IDF.ipynb：对使用TF-IDF方法得到的文档向量做分类预测
  - TF-IDF vector.py：将文本转成TF-IDF向量

- baseline_doc2vec:
  - Baseline：doc2vec.ipynb：对使用doc2vec方法得到的文档向量做分类预测
  - G-06-F-17_300d.model：300维的pre-trained模型
  - doc2vec.ipynb：使用genism库训练语料得到文档向量

- baseline_SIF:
  - SIF-master：SIF是一种根据word embedding得到sentences/paragraph/document特征表示的方法，由于其简单高效性，常被用做文本特性处理的baseline，原文是`Sanjeev Arora, Yingyu Liang, and Tengyu Ma. 2017. A simple but tough-to-beat baseline for sentence embeddings. In In Proceedings of ICLR`.
  - Baseline：SIF-MEAN.ipynb：对使用SIF方法得到的weighted embedding做分类预测
