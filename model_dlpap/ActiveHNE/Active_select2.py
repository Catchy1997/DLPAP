import numpy as np
from scipy import sparse as sp
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm
import time


def node_importance_degree(node_objects, all_adj, all_node_num):
    """
    calculate the node importance and node degree
    :param node_objects:
    :param all_adj:
    :param all_node_num:
    :return:
    """
    # all_adj is a csr_matrix
    importance = np.zeros(all_node_num)
    node_types = np.zeros(all_node_num)
    degree = np.zeros(all_node_num)
    all_node_type = len(node_objects)
    for index in range(len(node_objects)):
        nodes = node_objects[index]
        node_types[nodes] = index
    for index in range(all_node_num):
        idx_temp = all_adj[index, :]
        idx_neighbors = idx_temp.indices
        num_neighbors = len(idx_neighbors)
        degree[index] = num_neighbors
        node_type_temp = node_types[idx_neighbors]
        num_type_list = []
        for i in range(len(node_type_temp)):
            node_type = node_type_temp[i]
            if node_type not in num_type_list:
                num_type_list.append(node_type)
        num_neighbor_type = len(num_type_list)
        importance[index] = np.tanh(float(num_neighbors)/float(all_node_num) + float(num_neighbor_type)/float(all_node_type))
    return importance, degree


def centrality_select(pool_idx, importance, degree, all_adj, all_node_num, topb):
    """
    use network centrality (NC) to select nodes, i.e., select the topb nodes with the highest values of NC.
    :param pool_idx: the node index in the unlabeled pool.
    :param importance:
    :param degree:
    :param all_adj:
    :param all_node_num:
    :param topb: the batch size.
    :return:
    """
    centrality = np.zeros(all_node_num)
    # all_adj = sp.csc_matrix.todense(all_adj)

    for index in pool_idx:
        idx_temp = sp.csc_matrix.todense(all_adj[index, :]).tolist()[0]
        idx_temp[index] = 1
        idx_neighbors = np.where(np.array(idx_temp) > 0)[0]
        degree_temp = degree[idx_neighbors]
        importance_temp = importance[idx_neighbors]
        centrality[index] = np.dot(degree_temp, importance_temp)
    sort_idx = np.argsort(-centrality)
    sort_idx_topb = sort_idx[0:topb]
    return sort_idx_topb


def node_entropy(outputs, all_node_num):
    """
    calculate information entropy (IE) of nodes.
    :param outputs:
    :param all_node_num:
    :return:
    """
    entropy = np.zeros(all_node_num)
    for index in range(all_node_num):
        output = outputs[index, :]
        sum_output = sum(output)
        probs = np.array(output)/sum_output
        probs = probs * np.log2(probs)
        entropy[index] = -sum(probs)
    return entropy


def node_density(embeddings, all_node_num, class_num):
    """
    calculate information density (ID) of nodes.
    :param embeddings:
    :param all_node_num:
    :param class_num:
    :return:
    """
    cluster_num = class_num
    estimator = KMeans(n_clusters=cluster_num)
    estimator.fit(embeddings)
    centroids = estimator.cluster_centers_
    ed = euclidean_distances(embeddings, centroids)
    ed_score = np.min(ed, axis=1)
    density = 1.0/(1.0 + ed_score)
    return density


def active_select(outs, all_adj, pool_idx, all_node_num, topb, importance, degree, rewards, class_num, iter_num, dominates):
    """
    combine the three selection strategies to select the most valuable topb nodes with the highest scores.
    :param outs:
    :param all_adj:
    :param pool_idx: the node index of nodes in the unlabeled pool.
    :param all_node_num:
    :param topb: the batch size.
    :param importance:
    :param degree:
    :param rewards:
    :param class_num:
    :param iter_num: the number of iterations.
    :param dominates: a list with three elements, stores the number of nodes dominated by three arms (or selection strategies).
    :return:
    """
    # all_adj = sp.csc_matrix.todense(all_adj)
    # rewards = np.zeros(all_node_num)
    embeddings = outs[3]
    outputs = outs[4]
    centrality_rewards = rewards['centrality']
    entropy_rewards = rewards['entropy']
    density_rewards = rewards['density']

    if len(centrality_rewards) > 1:
        centrality_reward = (centrality_rewards[-1] + centrality_rewards[-2]) / 2.0
        entropy_reward = (entropy_rewards[-1] + entropy_rewards[-2]) / 2.0
        density_reward = (density_rewards[-1] + density_rewards[-2]) / 2.0
    else:
        centrality_reward = centrality_rewards[-1]
        entropy_reward = entropy_rewards[-1]
        density_reward = density_rewards[-1]
    if iter_num > 1:
        centrality_reward = centrality_reward + np.sqrt((3*np.log(iter_num)) / float(2*dominates['centrality']))
        entropy_reward = entropy_reward + np.sqrt((3 * np.log(iter_num)) / float(2 * dominates['entropy']))
        density_reward = density_reward + np.sqrt((3 * np.log(iter_num)) / float(2 * dominates['density']))

    centrality = np.zeros(all_node_num)
    info_entropy = np.zeros(all_node_num)
    info_density = np.zeros(all_node_num)

    entropy = node_entropy(outputs, all_node_num)
    density = node_density(embeddings, all_node_num, class_num)

    for index in tqdm(pool_idx, ncols=70):
        idx_temp = sp.csc_matrix.todense(all_adj[index, :]).tolist()[0]
        idx_temp[index] = 1
        idx_neighbors = np.where(np.array(idx_temp) > 0)[0]  # idx_neighbors ÎªarrayÀàÐÍ , ÁÚ¾Ó½ÚµãµÄË÷Òý

        degree_temp = degree[idx_neighbors]
        entropy_temp = entropy[idx_neighbors]
        density_temp = density[idx_neighbors]
        importance_temp = importance[idx_neighbors]
        # centrality[index] = np.dot(degree_temp, importance_temp)
        centrality[index] = degree[index]
        info_entropy[index] = np.dot(entropy_temp, importance_temp)
        info_density[index] = np.dot(density_temp, importance_temp)

    centrality_sort_idx = np.argsort(-centrality)
    centrality_sort_idx_topb = centrality_sort_idx[0:topb]
    entropy_sort_idx = np.argsort(-info_entropy)
    entropy_sort_idx_topb = entropy_sort_idx[0:topb]
    density_sort_idx = np.argsort(-info_density)
    density_sort_idx_topb = density_sort_idx[0:topb]
    rank = list(range(1, topb + 1))
    borda_count = topb - np.array(rank)

    all_idx = list(set(centrality_sort_idx_topb) | set(entropy_sort_idx_topb) | set(density_sort_idx_topb))

    if len(all_idx) == topb:
        dominates['centrality'] = dominates['centrality'] + topb
        dominates['entropy'] = dominates['entropy'] + topb
        dominates['density'] = dominates['density'] + topb
        return all_idx, all_idx, all_idx, all_idx,dominates

    centrality_sort_idx_topb = centrality_sort_idx_topb.tolist()
    entropy_sort_idx_topb = entropy_sort_idx_topb.tolist()
    density_sort_idx_topb = density_sort_idx_topb.tolist()
    rewards = np.zeros(len(all_idx))
    for i in tqdm(range(len(all_idx)), ncols=70):
        centrality_score = 0
        entropy_score = 0
        density_score = 0
        idx = all_idx[i]

        if idx in centrality_sort_idx_topb:
            find_idx = centrality_sort_idx_topb.index(idx)
            centrality_score = borda_count[find_idx]
        if idx in entropy_sort_idx_topb:
            find_idx = entropy_sort_idx_topb.index(idx)
            entropy_score = borda_count[find_idx]
        if idx in density_sort_idx_topb:
            find_idx = density_sort_idx_topb.index(idx)
            density_score = borda_count[find_idx]
        rewards[i] = centrality_reward * centrality_score + entropy_reward * entropy_score + density_reward * density_score
    rewards_sort_idx = np.argsort(-rewards)
    rewards_sort_idx_topb = rewards_sort_idx[0:topb + 1]
    all_idx = np.array(all_idx)
    idx_select = all_idx[rewards_sort_idx_topb].tolist()

    idx_select_centrality = list(set(idx_select).intersection(set(centrality_sort_idx_topb)))
    idx_select_entropy = list(set(idx_select).intersection(set(entropy_sort_idx_topb)))
    idx_select_density = list(set(idx_select).intersection(set(density_sort_idx_topb)))

    dominates['centrality'] = dominates['centrality'] + len(idx_select_centrality)
    dominates['entropy'] = dominates['entropy'] + len(idx_select_entropy)
    dominates['density'] = dominates['density'] + len(idx_select_density)

    return idx_select, idx_select_centrality, idx_select_entropy, idx_select_density, dominates
