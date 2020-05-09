import numpy as np
import scipy.sparse as sp


def parse_node_index_file(filename):
    """Parse node index file."""
    with open(filename, 'r') as file_to_read:
        readers = file_to_read.read().splitlines()
    index = []
    for i in range(len(readers)):
        temp = readers[i].strip()
        temp_list = temp.split('\t')
        index.append(int(temp_list[0].strip()))
    return index


def parse_edge_index_file(filename):
    """Parse edge index file."""
    with open(filename, 'r') as file_to_read:
        reader = file_to_read.read().splitlines()
    new_data = []
    old_data = []
    old_row = []
    old_col = []
    for index in range(len(reader)):
        if index > 2:
            line = reader[index]
            str_temp = line.strip().split("\t")
            row_index = int(str_temp[0].strip())
            col_index = int(str_temp[1].strip())
            old_row.append(row_index)
            old_col.append(col_index)
            old_data.append(int(1))
            if len(str_temp) > 2:
                new_data.append(float(str_temp[2].strip()))
            else:
                new_data.append(1)
    old_row1 = old_row + old_col
    old_col1 = old_col + old_row
    new_data1 = new_data + new_data
    old_data1 = old_data + old_data
    return old_row1, old_col1, new_data1, old_data1


def parse_label_index_file(filename):
    """Parse label index file."""
    with open(filename, 'r') as file_to_read:
        reader = file_to_read.read().splitlines()
    node_index = []
    label = []
    label_list = []
    label_list_new = []
    count = 0
    for index in range(len(reader)):
        line = reader[index]
        str_temp = line.strip().split("\t")
        node_index.append(int(str_temp[0].strip()))
        label_temp = int(str_temp[1].strip())
        label.append(label_temp)
        if label_temp not in label_list:
            label_list.append(label_temp)
            label_list_new.append(count)
            count = count + 1
    class_num = len(label_list)
    label_array = np.zeros((len(label), class_num))
    for index in range(len(label)):
        class_label = label[index]
        idx = label_list.index(class_label)
        class_label_new = label_list_new[idx]
        label_array[index][class_label_new] = 1
    return node_index, label_array, class_num


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from /data/ directory

    :param dataset_str: Dataset name
    :return:
           node_objects: a list, node_objects[i] stores a list of node index in a homogeneous network or bipartite network.
           feature: csr_matrix, feature matrix
           network_objects: a list, network_objects[i] stores a csr_matrix of adjacent matrix.
           all_y_index: a list, stores a list of node index of labeled nodes.
           all_y_label: a np.array, stores a label matrix.
           train_y_index: a list, stores a list of node index of train nodes.
           test_y_index: a list, stores a list of node index of test nodes.
           class_num: the number of class labels in the label space.
           all_node_num: the total number of nodes in the HIN.
           new_adj: stores the true weight values.
           old_adj: old_adj \in {0,1}.
    """
    node_index = []
    edge_index = []
    label_index = []
    train_label_index = []
    test_label_index = []
    if dataset_str == 'DBLP_four_area':
        node_index = ["author_dict.txt", "paper_dict.txt", "conf_dict.txt", "term_dict.txt"]
        edge_index = ["paper_author.txt", "paper_conf.txt", "paper_term.txt"]
        label_index = ["author_label.txt", "conf_label.txt", "paper_label.txt"]
        train_label_index = ["author_label_train_idx.txt", "conf_label_train_idx.txt", "paper_label_train_idx.txt"]
        test_label_index = ["author_label_test_idx.txt", "conf_label_test_idx.txt", "paper_label_test_idx.txt"]
    elif dataset_str == 'Cora':
        node_index = ["author_dict.txt", "paper_dict.txt", "term_dict.txt"]
        edge_index = ["PA.txt", "PT.txt", "PP.txt"]
        # edge_index = ["PA.txt", "PT.txt"]
        label_index = ["paper_label.txt"]
        train_label_index = ["paper_label_train_idx.txt"]
        test_label_index = ["paper_label_test_idx.txt"]
    elif dataset_str == 'MovieLens':
        node_index = ["directors_dict.txt", "movies_dict.txt", "tags_dict.txt", "users_dict.txt", "writers_dict.txt"]
        edge_index = ["movie_director.txt", "movie_tag.txt", "movie_writer.txt", "user_movie_rating.txt"]
        label_index = ["movie_genre.txt"]
        train_label_index = ["movie_genre_train_idx.txt"]
        test_label_index = ["movie_genre_test_idx.txt"]
    else:
        print("Data loading error: the loaded dataset was not found!")

    node_objects = []
    network_objects = []
    all_y_index = []
    all_y_label = []
    train_y_index = []
    test_y_index = []
    all_node_num = 0
    for index in range(len(node_index)):
        idx_reorder = parse_node_index_file("data/{}/{}".format(dataset_str, node_index[index]))
        all_node_num = all_node_num + len(idx_reorder)
        node_objects.append(idx_reorder)
    all_row = []
    all_col = []
    all_new_data = []
    all_old_data = []
    feature = sp.csr_matrix(sp.eye(all_node_num))
    for index in range(len(edge_index)):
        old_row, old_col, new_data, old_data = parse_edge_index_file("data/{}/{}".format(dataset_str, edge_index[index]))
        adj = sp.csr_matrix((new_data, (old_row, old_col)), shape=(all_node_num, all_node_num))
        network_objects.append(adj)
        all_row = all_row + old_row
        all_col = all_col + old_col
        all_old_data = all_old_data + old_data
        all_new_data = all_new_data + new_data
    """Construct the adjacent matrix of the whole HIN"""
    new_adj = sp.csr_matrix((all_new_data, (all_row, all_col)), shape=(all_node_num, all_node_num))
    old_adj = sp.csr_matrix((all_old_data, (all_row, all_col)), shape=(all_node_num, all_node_num))
    # ==================================================================================================
    class_num = 0
    for index in range(len(label_index)):
        labeled_node_index, label_array, class_num = \
            parse_label_index_file("data/{}/{}".format(dataset_str, label_index[index]))
        all_y_index = all_y_index + labeled_node_index
        label_array = label_array.tolist()
        all_y_label = all_y_label + label_array
    all_y_label = np.array(all_y_label)

    for index in range(len(train_label_index)):
        train_index = np.loadtxt("data/{}/{}".format(dataset_str, train_label_index[index]))
        """
        train_index: numpy.array
        """
        if len(train_y_index) > 0:
            train_y_index = np.hstack((train_y_index, train_index))
        else:
            train_y_index = train_index
    train_y_index = train_y_index.astype(int)
    
    for index in range(len(test_label_index)):
        test_index = np.loadtxt("data/{}/{}".format(dataset_str, test_label_index[index]))
        if len(test_y_index) > 0:
            test_y_index = np.hstack((test_y_index, test_index))
        else:
            test_y_index = test_index
    test_y_index = test_y_index.astype(int)
    
    return node_objects, feature, network_objects, all_y_index, all_y_label,\
           train_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -1).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    # return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return d_mat_inv_sqrt.dot(adj).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict




