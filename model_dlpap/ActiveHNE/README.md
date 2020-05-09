This program (ActiveHNE) implements an active learning for heterogeneous network embedding, as proposed in the following paper.
If you use it for scientific experiments, please cite this paper:
@inproceedings{Chen2019ActiveHNE,
  title={ActiveHNE: Active Heterogeneous Network Embedding},
  author={Chen, Xia and Yu, Guoxian and Wang, Jun and Domeniconi, Carlotta and Li, Zhao, and Zhang, Xiangliang},
  booktitle={IJCAI},
  pages={1-7},
  year={2019},
}

The codes of ActiveHNE are implemented and tested on Tensorflow 1.8.0 version.
The code of DHNE of ActiveHNE is prepared with reference to the original GCN source code. 

============== *** Requirements packages *** ============== 
tensorflow 
sklearn

============== *** Input data *** ==============

An example: 
       node_index = ["directors_dict.txt", "movies_dict.txt", "tags_dict.txt", "users_dict.txt", "writers_dict.txt"]
       edge_index = ["movie_director.txt", "movie_tag.txt", "movie_writer.txt", "user_movie_rating.txt"]
       label_index = ["movie_genre.txt"]
       train_label_index = ["movie_genre_train_idx.txt"]
       test_label_index = ["movie_genre_test_idx.txt"]

In order to use your own data, you have to first divide the original heterogeneous network into homogeneous networks and bipartite networks, and provide:

node_index: 
            the .txt files that store the indexes of nodes. Each file stores the indexes of nodes with respect to a particular node type (an HIN has more than one node types)  
            NOTE: all nodes in the HIN has a unique index. If there are N nodes in the HIN, the index of nodes ranges from [0, N-1]
edge_index:
           the .txt files that store the relationships (or edges) between nodes: each file stores the relationships between nodes in a particular divided network. 
           (an heterogeneous network has been divided into homogeneous networks and bipartite networks)
            an example:  "index_of_node_1"\t"index_of_node_2"\t"edge_weight" (for weighted networks) or "index_of_node_1"\t"index_of_node_2" (for unweighted networks)
			“index_of_node_1" and "index_of_node_2" represent the indexes of two nodes, respectively. NOTE: "\t" is used as the separator, 	   
label_index:
		   a .txt file that stores the indexes of all labeled nodes. 
           NOTE: The labeled nodes here just mean the nodes with labels in the dataset, not the labeled nodes used for semi-supervised model training after data division. 
train_label_index: 
                 a .txt file that stores the indexes of 50% labeled nodes (randomly selected from "label_index"). It includs the 25% nodes of training set and the 25% nodes of validation set.
                 an example:  "index_of_node"\t"label"				 
test_label_index:
                 a .txt file that stores the indexes of 50% labeled nodes (randomly selected from "label_index") used for testing (test set).
                 an example:  "index_of_node"\t"label"					 
			   

============== *** Run the Program *** ==============
Command: 
python ActiveHNE.py 

