import numpy as np
from scipy import sparse as sp


def measure_rewards(outs_new, outs_old, rewards, adj, idx_select, idx_select_centrality, idx_select_entropy, idx_select_density):
    """
    :param outs_new: the new outputs from DHNE
    :param outs_old: the old outputs from DHNE
    :param rewards:  a 2-dimensional list, stores the previous rewards of three arms (or selection strategies).
    :param adj: the adjacent matrix.
    :param idx_select: a list, stores the nodes index of all selected nodes.
    :param idx_select_centrality: a list, stores the nodes index of the selected nodes by centrality (NC).
    :param idx_select_entropy: a list, stores the nodes index of the selected nodes by entropy (CIE).
    :param idx_select_density: a list, stores the nodes index of the selected nodes by density (CID).
    :return:
         rewards:  a 2-dimensional list, stores the previous rewards and the current rewards of three arms (or selection strategies).
    """
    # adj = sp.csc_matrix.todense(adj)
    outputs_new = outs_new[4]
    outputs_old = outs_old[4]
    distance_sum = 0.0
    distance_sum_centrality = 0.0
    distance_sum_entropy = 0.0
    distance_sum_density = 0.0
    for index in range(len(idx_select)):
        node_idx = idx_select[index]
        idx_temp = sp.csc_matrix.todense(adj[node_idx, :]).tolist()[0]
        idx_neighbors = np.where(np.array(idx_temp) > 0)[0]
        output_neighbor_old = outputs_old[idx_neighbors, :]
        output_neighbor_new = outputs_new[idx_neighbors, :]
        for i in range(len(idx_neighbors)):
            distance = np.sqrt(np.sum(np.square(output_neighbor_new[i, :], output_neighbor_old[i, :])))
            # distance = euclidean_distances(output_neighbor_new[i, :], output_neighbor_old[i, :])
            distance_sum += distance

    for index in range(len(idx_select_centrality)):
        node_idx = idx_select_centrality[index]
        idx_temp = sp.csc_matrix.todense(adj[node_idx, :]).tolist()[0]
        idx_neighbors = np.where(np.array(idx_temp) > 0)[0]
        output_neighbor_old = outputs_old[idx_neighbors, :]
        output_neighbor_new = outputs_new[idx_neighbors, :]
        for i in range(len(idx_neighbors)):
            distance = np.sqrt(np.sum(np.square(output_neighbor_new[i, :], output_neighbor_old[i, :])))
            distance_sum_centrality += distance

    for index in range(len(idx_select_entropy)):
        node_idx = idx_select_entropy[index]
        idx_temp = sp.csc_matrix.todense(adj[node_idx, :]).tolist()[0]
        idx_neighbors = np.where(np.array(idx_temp) > 0)[0]
        output_neighbor_old = outputs_old[idx_neighbors, :]
        output_neighbor_new = outputs_new[idx_neighbors, :]
        for i in range(len(idx_neighbors)):
            distance = np.sqrt(np.sum(np.square(output_neighbor_new[i, :], output_neighbor_old[i, :])))
            distance_sum_entropy += distance

    for index in range(len(idx_select_density)):
        node_idx = idx_select_density[index]
        idx_temp = sp.csc_matrix.todense(adj[node_idx, :]).tolist()[0]
        idx_neighbors = np.where(np.array(idx_temp) > 0)[0]
        output_neighbor_old = outputs_old[idx_neighbors, :]
        output_neighbor_new = outputs_new[idx_neighbors, :]
        for i in range(len(idx_neighbors)):
            distance = np.sqrt(np.sum(np.square(output_neighbor_new[i, :], output_neighbor_old[i, :])))
            distance_sum_density += distance

    reward_centrality = distance_sum_centrality/float(distance_sum)
    reward_entropy = distance_sum_entropy / float(distance_sum)
    reward_density = distance_sum_density / float(distance_sum)

    rewards_centrality = rewards['centrality']
    rewards_entropy = rewards['entropy']
    rewards_density = rewards['density']

    rewards_centrality.append(reward_centrality)
    rewards_entropy.append(reward_entropy)
    rewards_density.append(reward_density)
    rewards['centrality'] = rewards_centrality
    rewards['entropy'] = rewards_entropy
    rewards['density'] = rewards_density

    return rewards
