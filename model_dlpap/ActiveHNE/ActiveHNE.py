# -*- coding: utf-8 -*-
import time
import tensorflow as tf
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

from utils2 import *
from models import DHNE
import random
from Active_select2 import *
from rewards import *

def plot_history(loss, acc, val_loss, val_acc, epochNum, epoch, run):
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/ActiveHNE/data/pic-"+str(epochNum)+"-"+str(run)+"-"+str(epoch)+".png")

# Define model training function
def DHNE_train(epochNum, run, iter_num, y_train1, train_mask1, y_val1, val_mask1, y_test1, test_mask1):
    model_t = time.time()
    # Create model
    model = DHNE(placeholders, input_dim=features[2][1], logging=True)
    # Initialize session
    sess = tf.Session()
    # Init variables
    sess.run(tf.global_variables_initializer())
    # Train model
    outs_train = []
    train_loss_list = []
    train_acc_list = []
    val_acc_list = []
    val_loss_list = []
    count = 0
    for epoch in range(epochNum):
        # epoch_t = time.time()

        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train1, train_mask1, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Training step
        outs_train = sess.run([model.opt_op, model.loss, model.accuracy, model.outputs, model.predict(), model.vector,], feed_dict=feed_dict)
        train_loss_list.append(outs_train[1])
        train_acc_list.append(outs_train[2])
        # Validation
        feed_dict_val = construct_feed_dict(features, support, y_val1, val_mask1, placeholders)
        outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
        val_loss_list.append(outs_val[0])
        val_acc_list.append(outs_val[1])
        # val_cost0 = outs_val[0]
        # val_acc0 = outs_val[1]
        # duration0 = time.time() - model_t
        # print("Validation: epoch=" + str(epoch), "  Test set results:", "cost=", "{:.5f}".format(val_cost0),
        # "accuracy=", "{:.5f}".format(val_acc0), "time=", "{:.5f}".format(duration0))
        if epoch > 0:
        	val_loss_entropy = val_loss_list[-1] - val_loss_list[-2]
        	if val_loss_entropy > 0:
        		count = count + 1
        	if count >= 10:
        		break
    # plot_history(train_loss_list, train_acc_list, val_loss_list, val_acc_list, epochNum, epoch, run)

    # Testing
    feed_dict_test = construct_feed_dict(features, support, y_test1, test_mask1, placeholders)
    outs_test = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_test)
    test_cost0 = outs_test[0]
    test_acc0 = outs_test[1]
    duration0 = time.time() - model_t
    # print("Testing: epoch=" + str(epoch), "  Test set results:", "cost=", "{:.5f}".format(test_cost0),
    # "accuracy=", "{:.5f}".format(test_acc0), "time=", "{:.5f}".format(duration0))
    # print("Optimization Finished!")
    # print("Optimization Finished!")
        
    return test_cost0, test_acc0, duration0, outs_train, count

if __name__ == '__main__':
    # Set random seed
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)

    # Settings
    flags = tf.app.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('dataset', 'Cora', 'Dataset string.')  # 'MovieLens', 'Cora', 'DBLP_four_area'
    flags.DEFINE_string('model', 'DHNE', 'Model string.')
    flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
    flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
    flags.DEFINE_integer('hidden1', 32, 'Number of units in hidden layer 1.')
    flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
    flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
    dataset_arr = FLAGS.dataset.split('|')

    # Load data
    data_str = dataset_arr[0]
    node_objects, features, network_objects, all_y_index, all_y_label, pool_y_index, test_y_index, class_num, all_node_num, new_adj, old_adj = load_data(data_str)
    importance, degree = node_importance_degree(node_objects, old_adj, all_node_num)
        
    # Some preprocessing
    # features = preprocess_features(features)
    features = sparse_to_tuple(features)
    # support = [preprocess_adj(new_adj)]
    support = []
    for index in range(len(network_objects)):
        adj = network_objects[index]
        support.append(preprocess_adj(adj))
    num_supports = len(support)

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, class_num)),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    #  Active Learning
    num_train_nodes = len(pool_y_index[0])
    batch = 2560 # ---------------------------------------------------------------
    round_num = len(pool_y_index)
    num_pool_nodes = int(num_train_nodes / 2)
    # num_pool_nodes = 200
    maxIter = int(num_pool_nodes / batch)
    if maxIter > 40:
        maxIter = 15 # ---------------------------------------------------------------
    print("Iteraion times:" + str(maxIter))
    results = []
    model_times = []
    select_times = []
    rewards_centrality = []
    rewards_entropy = []
    rewards_density = []

    # begin train
    roundNum = 1 # ---------------------------------------------------------------
    acc_list = []
    time_list = []
    for run in range(roundNum):
        result_temp = []
        model_time_temp = []
        select_time_temp = []

        y_all = np.zeros((all_node_num, class_num))
        y_all[all_y_index, :] = all_y_label

        val_idx = pool_y_index[run][num_pool_nodes:num_train_nodes]
        val_mask = sample_mask(val_idx, all_node_num)
        y_val = np.zeros((all_node_num, class_num))
        y_val[val_mask, :] = y_all[val_mask, :]
        pool_idx = pool_y_index[run][0:num_train_nodes]
        test_idx = test_y_index[run]
        pool_mask = sample_mask(pool_idx, all_node_num)
        test_mask = sample_mask(test_idx, all_node_num)
        y_pool = np.zeros((all_node_num, class_num))
        y_test = np.zeros((all_node_num, class_num))
        y_pool[pool_mask, :] = y_all[pool_mask, :]
        y_test[test_mask, :] = y_all[test_mask, :]
        pool_idx = pool_idx.tolist()
        random.shuffle(pool_idx)
        outs_train = []
        train_idx = []
        outs_new = []
        outs_old = []
        rewards = dict()
        reward_centrality = []
        reward_entropy = []
        reward_density = []
        rewards['centrality'] = reward_centrality
        rewards['entropy'] = reward_entropy
        rewards['density'] = reward_density
        idx_select = []
        idx_select_centrality = []
        idx_select_entropy = []
        idx_select_density = []
        dominates = dict()
        dominates['centrality'] = 0
        dominates['entropy'] = 0
        dominates['density'] = 0
        
        for iter_num in range(maxIter):
            select_t = time.time()
            if iter_num == 0:
                idx_select = pool_idx[0:batch]
            else:
                idx_select, idx_select_centrality, idx_select_entropy, idx_select_density, dominates = active_select(outs_train, old_adj, pool_idx, all_node_num, batch, importance, degree, rewards, class_num, iter_num, dominates)
            select_duration = time.time() - select_t

            pool_idx = list(set(pool_idx) - set(idx_select))
            train_idx = train_idx + idx_select
            train_mask = sample_mask(train_idx, all_node_num)
            y_train = np.zeros((all_node_num, class_num))
            y_train[train_mask, :] = y_all[train_mask, :]
            test_cost, test_acc, model_duration, outs_train, count = DHNE_train(FLAGS.epochs, run, iter_num, y_train, train_mask, y_val, val_mask, y_test, test_mask)
            acc_list.append(test_acc)
            time_list.append(model_duration)

            # write down embedding
            vector = outs_train[5]*1000
            vector = vector.tolist()
            for i in range(0, 7769):
                with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/ActiveHNE/data/"+str(run)+"-"+str(iter_num)+".txt", 'a') as f:
                    f.write(str(vector[i]))
                    f.write('\n')
            
            print("round=" + str(run), " iter=" + str(iter_num), "  Test set results:", "cost=", "{:.5f}".format(test_cost), "accuracy=", "{:.5f}".format(test_acc), "model_duration=", "{:.5f}".format(model_duration), "select_duration=", "{:.5f}".format(select_duration))

            with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/ActiveHNE/data/acc_result.txt", 'a') as f:
            	f.write("round="+str(run)+"\titer="+str(iter_num))
            	f.write("\tcost={:.5f}".format(test_cost))
            	f.write("\taccuracy={:.5f}".format(test_acc))
            	f.write("\tmodel_duration={:.5f}".format(model_duration))
            	f.write("\tselect_duration={:.5f}".format(select_duration))
            	f.write("\n")

            outs_old = outs_new
            outs_new = outs_train
            if iter_num == 0:
                reward_centrality = rewards['centrality']
                reward_entropy = rewards['entropy']
                reward_density = rewards['density']

                reward_centrality.append(1)
                reward_entropy.append(1)
                reward_density.append(1)
                rewards['centrality'] = reward_centrality
                rewards['entropy'] = reward_entropy
                rewards['density'] = reward_density
            else:
                rewards = measure_rewards(outs_new, outs_old, rewards, old_adj, idx_select, idx_select_centrality, idx_select_entropy, idx_select_density)

    with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/ActiveHNE/data/parameter.txt", 'a') as f:
        text = "weight_decay="+str(FLAGS.weight_decay)+" dropout="+str(FLAGS.dropout)+" epochs="+str(FLAGS.epochs)+" hidden1="+str(FLAGS.hidden1)+" learning_rate="+str(FLAGS.learning_rate)+" batch="+str(batch)+" maxIter="+str(maxIter)+" roundNum="+str(roundNum)+" early_stop="+str(count)
        f.write(text)
        f.write('\n')
        f.write("acc:  {:.4f}".format(np.mean(acc_list)))
        f.write('\n')
        f.write("model time:  {:.4f}".format(np.mean(time_list)))
        f.write('\t')
        f.write("select time:  {:.4f}".format(select_duration))
        f.write('\n')
    print("END")
