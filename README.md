# DLPAP
Code for the paper `Deep Learning for Patent Application: The Fusion of Network and Text Embedding`

### 项目文件说明：
- data_process：数据处理
   - class_count.py：提取patent application中的cpc分类信息，按照每一年进行统计
   - get_appID_from_citaion.py：`location`和`application_id`字段的转换
   - index_build.py：遍历patent application文本，建立patent-inventor、patent-assignee之间的关系作为索引，按照每一年进行统计
   - split_citations.py：对PatentsView中统计的citations信息按照每一年进行划分和统计
   - xmlfile_deal.py：对patent application的原始zip文件进行数据预处理

- sample：样本数据采集
  - 1get_patent_random.py：随机采集不同规模的patent数量 5000/10000/20000
  - 2get_text.py：提取专利文本信息
  - 3get_network.py：提取专利网络信息
  - 4get_history.py：提取历史专利信息
  - 5get_parent.py：提取原申请（parent patent）专利
  - 6get_history_result.py：提取标签结果
  - get_patent_by_class.py
  - node.ipynb：节点数据的处理，建立唯一ID的对应关系并根据时间构建动态异构网络

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
  - doc2vec.ipynb：使用genism库训练语料得到文档向量

- baseline_SIF:
  - SIF-master：SIF是一种根据word embedding得到sentences/paragraph/document特征表示的方法，由于其简单高效性，常被用做文本特性处理的baseline，原文是`Sanjeev Arora, Yingyu Liang, and Tengyu Ma. 2017. A simple but tough-to-beat baseline for sentence embeddings. In In Proceedings of ICLR`.
  - Baseline：SIF-MEAN.ipynb：对使用SIF方法得到的weighted embedding做分类预测
