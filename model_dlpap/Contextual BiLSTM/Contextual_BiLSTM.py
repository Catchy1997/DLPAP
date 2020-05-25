import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import layers, models, optimizers
from keras.layers.merge import concatenate
from keras.metrics import categorical_accuracy
from keras.utils import to_categorical, np_utils
from keras import backend as K

import tensorflow as tf

# 在sigmoid激活的前提下，计算权重交叉熵
def weighted_cross_entropy(beta):
    def convert_to_logits(y_pred):
        # see https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/python/keras/backend.py#L3525
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        return tf.log(y_pred / (1 - y_pred))

    def loss(y_true, y_pred):
        """
        y_true shape need be (None,1)
        y_pred need be compute after sigmoid
        """
        y_pred = convert_to_logits(y_pred)
        loss = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, targets=y_true, pos_weight=beta)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

    return loss

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())

    return auc

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.
        Only computes a batch-wise average of recall.
        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall
 
    def precision(y_true, y_pred):
        """Precision metric.
        Only computes a batch-wise average of precision.
        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def balanced_recall(y_true, y_pred):
    """
    Computes the average per-column recall metric
    for a multi-class classification problem
    """ 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)), axis=0)  
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)), axis=0)   
    recall = true_positives / (possible_positives + K.epsilon())    
    balanced_recall = K.mean(recall)
    return balanced_recall

def plot_history(name, history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
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
    plt.savefig("./"+name+"/"+name+".png")

def unimodal(text_type):    
    input_layer = layers.Input((SEQ_LEN, ))
    embedding_layer = layers.Embedding(input_dim=len(VOCAB_SIZE) + 1,
                                output_dim=EMBEDDING_DIM, weights=[embedding_matrix],
                                mask_zero=True, trainable=False)(input_layer)

    # dropout0_layer = layers.SpatialDropout1D(0.1)(embedding_layer)
    lstm_layer = layers.Bidirectional(layers.LSTM(300, activation='tanh', dropout=0.4, return_sequences=False))(embedding_layer)
    normalize_layer = layers.BatchNormalization(momentum=0.5, epsilon=1e-06)(lstm_layer)
    dense_layer = layers.Dense(72, activation='tanh')(normalize_layer)
    normalize_layer = layers.BatchNormalization(momentum=0.5, epsilon=1e-06)(dense_layer)
    dropout2_layer = layers.Dropout(0.6)(normalize_layer)
    output_layer = layers.Dense(2, activation='softmax')(dropout2_layer)
    
    model = models.Model(inputs=input_layer, outputs=output_layer)
    # model.compile(optimizer=optimizers.Adadelta(lr=0.3), loss=weighted_cross_entropy(10), metrics=['acc'])
    model.compile(optimizer=optimizers.Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

    with open("./"+text_type+"/"+text_type+"_result.txt", 'a') as f:
        f.write("Patameters:\n")
        f.write("SpatialDropout1D=0\t")
        f.write("BiLSTM node=300, activation=tanh, dropout=0.4\t")
        f.write("BatchNormalization=0.5\t")
        f.write("Dense nodes=72, activation=tanh\t")
        f.write("BatchNormalization=0.5\t")
        f.write("Dropout=0.6\t")
        f.write("learning rate=1")
        f.write("\n---------------------\n")

    return model

def multimodal():
    input_layer = layers.Input((x_fusion_train.shape[1], ))

    input_layer_1 = layers.Lambda(lambda input_layer:K.expand_dims(input_layer, axis=-1))(input_layer)
    lstm_layer = layers.Bidirectional(layers.LSTM(144, activation='tanh', dropout=0.4, return_sequences=False))(input_layer_1)
    normalize_layer = layers.BatchNormalization(momentum=0.9, epsilon=1e-06)(lstm_layer)
    dense_layer = layers.Dense(72, activation="tanh")(normalize_layer)
    normalize_layer = layers.BatchNormalization(momentum=0.9, epsilon=1e-06)(dense_layer)
    dropout2_layer = layers.Dropout(0.6)(normalize_layer)
    output_layer = layers.Dense(2, activation="softmax")(dropout2_layer)  
    
    model = models.Model(inputs=input_layer, output=output_layer)
    model.compile(optimizer=optimizers.Adadelta(lr=0.2), loss='categorical_crossentropy', metrics=['accuracy'])
    
    # with open("./fusion/fusion_result.txt", 'a') as f:
    #     f.write("Patameters:\n")
    #     f.write("BiLSTM node=144, activation=tanh, dropout=0.4\t")
    #     f.write("BatchNormalization=0.9\t")
    #     f.write("Dense nodes=72, activation=tanh\t")
    #     f.write("BatchNormalization=0.9\t")
    #     f.write("Dropout=0.6\t")
    #     f.write("Dense nodes=2, activation=softmax")
    #     f.write("\n---------------------\n")
    
    return model

def data_process(text_file, result_file, text_type):
    data = pd.read_excel(text_file,encoding='utf-8')
    result_data = pd.read_excel(result_file, encoding='utf-8')

    text_content = data[text_type + "_final"]

    x_train = text_content.iloc[:6215]
    x_valid = text_content.iloc[6215:6992]
    x_test = text_content.iloc[6992:]

    train_target = np_utils.to_categorical(result_data[['result']], 2)
    y_ints = [y.argmax() for y in train_target]
    cw = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)

    y_category_train = train_target[:6215]
    y_category_valid = train_target[6215:6992]
    y_category_test = train_target[6992:]

    return text_content, x_train, x_valid, x_test, cw, y_category_train, y_category_valid, y_category_test

def load_obj(embedding_file):
    with open(embedding_file, 'rb') as f:
        return pickle.load(f)

def get_embedding(text_type):
    token = text.Tokenizer()
    token.fit_on_texts(text_content)
    word_index = token.word_index
    if text_type == "abstract":
        maxlen = 140
    if text_type == "claims":
        maxlen = 1400
    x_seq = sequence.pad_sequences(token.texts_to_sequences(text_content), maxlen=maxlen)
    x_train_seq = sequence.pad_sequences(token.texts_to_sequences(x_train), maxlen=maxlen)
    x_valid_seq = sequence.pad_sequences(token.texts_to_sequences(x_valid), maxlen=maxlen)
    x_test_seq = sequence.pad_sequences(token.texts_to_sequences(x_test), maxlen=maxlen)

    embedding_matrix = np.zeros((len(word_index) + 1, 300))
    for word, i in tqdm(word_index.items(), ncols=70):
        word = word.strip('\'')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    # 得到embedding的单词占比
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix, axis=1))
    cover_per = nonzero_elements / (len(word_index) + 1)
    print("cover: {:.4f}".format(cover_per))

    return x_seq, x_train_seq, x_valid_seq, x_test_seq, embedding_matrix, word_index

if __name__ == '__main__':
    Epoch = 50
    BATCH_SIZE = 128

    # 准备文本
    text_file = "/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/after_process.xlsx"
    result_file = "/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/2010_result.xlsx"
    
    # 加载词向量
    embedding_file = "./embedding.pkl"
    embeddings_index = load_obj(embedding_file)
    
    # get unimodal
    # text_type_list = ["abstract"]
    # for text_type in text_type_list:
    #     text_content, x_train, x_valid, x_test, cw, y_category_train, y_category_valid, y_category_test = data_process(text_file, result_file, text_type)
    #     x_seq, x_train_seq, x_valid_seq, x_test_seq, embedding_matrix, word_index = get_embedding(text_type)

    #     EMBEDDING_DIM = embedding_matrix.shape[1]
    #     SEQ_LEN = x_train_seq.shape[1]
    #     VOCAB_SIZE = word_index

    #     model = unimodal(text_type)
    #     # print(model.summary())
        
    #     model_name = text_type + "-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"
    #     filepath = "./" + text_type + "/" + model_name
    #     my_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')
    #     # my_callbacks = [EarlyStopping(monitor='val_loss', mode='min', patience=5)]
    #     # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
    #     history = model.fit(x_train_seq, y_category_train, validation_data=(x_valid_seq, y_category_valid), 
    #                         verbose=1, epochs=Epoch, batch_size=BATCH_SIZE, shuffle=True, callbacks=[my_checkpoint])
    #     plot_history(text_type, history)

    #     y_pred = model.predict(np.array(x_test_seq), batch_size=BATCH_SIZE, verbose=0)
    #     for i in range(len(y_pred)):
    #          max_value=max(y_pred[i])
    #          for j in range(len(y_pred[i])):
    #              if max_value==y_pred[i][j]:
    #                  y_pred[i][j]=1
    #              else:
    #                  y_pred[i][j]=0
    #     report = classification_report(y_category_test, y_pred, digits=4)
    #     print(report)
    #     p = re.compile('  |\n', re.S)
    #     report = p.sub(' ', report)
    #     metrics_content = re.findall("([\d]{1}\.[\d]{4})    777", report)
    #     with open("./"+text_type+"/"+text_type+"_result.txt", 'a') as f:
    #          f.write("Result:\n")
    #          f.write("balanced=yes"+"\tacc=" + metrics_content[0])
    #          f.write("\tmacro=" + metrics_content[1])
    #          f.write("\tweighted=" + metrics_content[2])
    #          f.write("\n---------------------\n")
    
    # get multimodal
    text_type_list = ["abstract", "claims"]
    for text_type in text_type_list:
        text_content, x_train, x_valid, x_test, cw, y_category_train, y_category_valid, y_category_test = data_process(text_file, result_file, text_type)
        x_seq, x_train_seq, x_valid_seq, x_test_seq, embedding_matrix, word_index = get_embedding(text_type)

        if text_type == "abstract":
            best_abs_model_filepath = "./abstract/abstract-ep008-loss0.623-val_loss0.611.h5"
            abstract_model = models.load_model(best_abs_model_filepath)
            abs_layer_model = models.Model(inputs=[abstract_model.layers[0].input],outputs=[abstract_model.layers[-4].output])
            abs_intermediate_output = abs_layer_model.predict(x_seq, batch_size=BATCH_SIZE, verbose=1)
        if text_type == "claims":
            best_claims_model_filepath = "./claims/claims-ep007-loss0.627-val_loss0.615.h5"
            claims_model = models.load_model(best_claims_model_filepath)
            claims_layer_model = models.Model(inputs=[claims_model.layers[0].input],outputs=[claims_model.layers[-4].output])
            claims_intermediate_output = claims_layer_model.predict(x_seq, batch_size=BATCH_SIZE, verbose=1)

    fusion_data = np.concatenate((abs_intermediate_output,claims_intermediate_output), axis=1)

    # x_fusion_train = fusion_data[:6215]
    # x_fusion_valid = fusion_data[6215:6992]
    # x_fusion_test = fusion_data[6992:]

    # fusion_model = multimodal()
    # # print(fusion_model.summary())

    # fusion_model_name = "fusion-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"
    # filepath = './fusion/' + fusion_model_name
    # fusion_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='min')    
    # fusion_history = fusion_model.fit(x_fusion_train, y_category_train, 
    #                                  validation_data=(x_fusion_valid, y_category_valid),
    #                                  verbose=1, epochs=Epoch, batch_size=BATCH_SIZE, shuffle=True, 
    #                                  callbacks=[fusion_checkpoint])
    # plot_history("fusion", fusion_history)

    # y_pred = fusion_model.predict(x_fusion_test, batch_size=BATCH_SIZE, verbose=0)
    # for i in range(len(y_pred)):
    #      max_value=max(y_pred[i])
    #      for j in range(len(y_pred[i])):
    #          if max_value==y_pred[i][j]:
    #              y_pred[i][j]=1
    #          else:
    #              y_pred[i][j]=0
    # report = classification_report(y_category_test, y_pred, digits=4)
    # print(report)
    # p = re.compile('  |\n', re.S)
    # report = p.sub(' ', report)
    # metrics_content = re.findall("([\d]{1}\.[\d]{4})    777", report)
    # # with open("./fusion/fusion_result.txt", 'a') as f:
    # #      f.write("Result:\n")
    # #      f.write("acc=" + metrics_content[0])
    # #      f.write("\tmacro=" + metrics_content[1])
    # #      f.write("\tweighted=" + metrics_content[2])
    # #      f.write("\n---------------------\n")
    
    # model_list = ["fusion-ep019-loss0.719-val_loss0.616.h5", "fusion-ep039-loss0.610-val_loss0.616.h5", "fusion-ep046-loss0.608-val_loss0.612.h5", "fusion-ep047-loss0.602-val_loss0.610.h5"]
    # for modelname in model_list:
    #     best_fusion_model_filepath = "./fusion/"+modelname
    #     fusion_model = models.load_model(best_fusion_model_filepath)
    #     fusion_layer_model = models.Model(inputs=[fusion_model.layers[0].input],outputs=[fusion_model.layers[-4].output])
    #     fusion_intermediate_output = fusion_layer_model.predict(fusion_data, batch_size=BATCH_SIZE, verbose=1)
    #     df = pd.DataFrame(fusion_intermediate_output)
    #     df.to_excel(best_fusion_model_filepath+".xlsx")