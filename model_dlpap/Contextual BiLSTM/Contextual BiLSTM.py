import pandas as pd
import numpy as np
import re
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

from keras.preprocessing import text, sequence
from keras.callbacks import ModelCheckpoint
from keras import layers, models, optimizers
from keras.layers import LSTM
from keras.layers.merge import concatenate
from keras.utils import to_categorical
from keras.utils import np_utils

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
    plt.savefig("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/"+name+".png")

def fusion_network():
    # for abstract
    input_layer_abstract = layers.Input((SEQ_LEN_abstract, ))
    embedding_layer_abstract = layers.Embedding(input_dim=len(VOCAB_SIZE_abstract) + 1,
                                       output_dim=EMBEDDING_DIM,
                                       weights=[embedding_matrix_abstract],
                                       mask_zero=True,
                                       trainable=False)(input_layer_abstract)    
    # 1.GlobalMaxPool1D
    # pool_layer_abstract = layers.GlobalMaxPool1D()(embedding_layer_abstract)
    # 2.Convolution
    # conv_layer_abstract = layers.Conv1D(128, 5, activation="relu")(embedding_layer_abstract)
    # pool_layer_abstract = layers.GlobalMaxPool1D()(conv_layer_abstract)
    # 3.Bidirectional
    lstm_layer_abstract = layers.Bidirectional(LSTM(300, return_sequences=True))(embedding_layer_abstract)
    
    # dense_layer_abstract = layers.Dense(72, activation="relu")(lstm_layer_abstract)
    # dropout_layer_abstract = layers.Dropout(0.5)(dense_layer_abstract)
    # output_layer_abstract = layers.Dense(2, activation="softmax")(dropout_layer_abstract)
    # abstract_model = models.Model(inputs=input_layer_abstract, outputs=output_layer_abstract)
    # abstract_model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    # with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/abstract_result.txt", 'a') as f:
    #     f.write("Patameters:\n")
    #     f.write("Embedding mask_zero=True\t")
    #     f.write("Embedding trainable=False\t")
    #     f.write("BiLSTM+Dense\t")
    #     f.write("LSTM nodes=300\t")
    #     f.write("Dense nodes=72\t")
    #     f.write("Dropout=0.5")
    #     f.write("\n")

    # for claims
    input_layer_claims = layers.Input((SEQ_LEN_claims, ))
    embedding_layer_claims = layers.Embedding(input_dim=len(VOCAB_SIZE_claims) + 1,
                                       output_dim=EMBEDDING_DIM,
                                       weights=[embedding_matrix_claims],
                                       mask_zero=True,
                                       trainable=False)(input_layer_claims)
    lstm_layer_claims = layers.Bidirectional(LSTM(300, return_sequences=True))(embedding_layer_claims)
    
    # dense_layer_claims = layers.Dense(72, activation="relu")(lstm_layer_claims)
    # dropout_layer_claims = layers.Dropout(0.5)(dense_layer_claims)
    # output_layer_claims = layers.Dense(2, activation="softmax")(dropout_layer_claims)
    # claims_model = models.Model(inputs=input_layer_claims, outputs=output_layer_claims)
    # claims_model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    # with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/claims_result.txt", 'a') as f:
    #     f.write("Patameters:\n")
    #     f.write("Embedding mask_zero=True\t")
    #     f.write("Embedding trainable=False\t")
    #     f.write("BiLSTM+Dense\t")
    #     f.write("LSTM nodes=300\t")
    #     f.write("Dense nodes=72\t")
    #     f.write("Dropout=0.5")
    #     f.write("\n")

    # for fusion
    # 1.LSTM
    lstm_fusion_layer = layers.LSTM(600, return_sequences=False)(concatenate([lstm_layer_abstract, lstm_layer_claims], axis=1))
    # 2.BiLSTM
    # 3.Dense    
    # fusion_dropout_1 = layers.Dropout(0.4)(lstm_fusion_layer)
    
    dense_layer_fusion = layers.Dense(72, activation="relu")(lstm_fusion_layer)
    dropout_layer_fusion = layers.Dropout(0.4)(dense_layer_fusion)
    output_layer_fusion = layers.Dense(2, activation="softmax")(dropout_layer_fusion)  
    fusion_model = models.Model(inputs=[input_layer_abstract, input_layer_claims], output=output_layer_fusion)
    fusion_model.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/fusion_result.txt", 'a') as f:
        f.write("Patameters:\n")
        f.write("LSTM+Dense\t")
        f.write("LSTM nodes=600\t")
        f.write("Dense nodes=72\t")
        f.write("Dropout=0.4")
        f.write("\n")
    
    # return abstract_model, claims_model, fusion_model
    return fusion_model

if __name__ == '__main__':
    # 0.设置参数
    Epoch = 10
    BATCH_SIZE = 256

    # 1.准备文本
    filename = "/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/after_process.xlsx"
    data = pd.read_excel(filename,encoding='utf-8')

    # 2.划分训练集和测试集
    abstract = data['abstract_final']
    claims = data['claims_final']
    result = data['result']

    x_abs_train = abstract.iloc[:6215]
    x_abs_valid = abstract.iloc[6215:6992]
    x_abs_test = abstract.iloc[6992:]

    x_claims_train = claims.iloc[:6215]
    x_claims_valid = claims.iloc[6215:6992]
    x_claims_test = claims.iloc[6992:]

    result = data['result']
    train_target = np_utils.to_categorical(data[['result']], 2)
    y_ints = [y.argmax() for y in train_target]
    cw = class_weight.compute_class_weight('balanced', np.unique(y_ints), y_ints)

    y_binary_train = result[:6215]
    y_binary_valid = result[6215:6992]
    y_binary_test = result[6992:]
    y_category_train = train_target[:6215]
    y_category_valid = train_target[6215:6992]
    y_category_test = train_target[6992:]

    # 3.以词向量为特征
    embeddings_index = {}
    for line in tqdm(open('/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/glove.42B.300d.word2vec.txt', 'r', encoding='UTF-8')):
        values= line.split()
        embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')

    abstract_token = text.Tokenizer()
    abstract_token.fit_on_texts(abstract)
    abstract_word_index = abstract_token.word_index

    x_abs_train_seq = sequence.pad_sequences(abstract_token.texts_to_sequences(x_abs_train), maxlen=140)
    x_abs_valid_seq = sequence.pad_sequences(abstract_token.texts_to_sequences(x_abs_valid), maxlen=140)
    x_abs_test_seq = sequence.pad_sequences(abstract_token.texts_to_sequences(x_abs_test), maxlen=140)

    embedding_matrix_abstract = np.zeros((len(abstract_word_index) + 1, 300)) # 300是词向量的维度,+1 is because the matrix indices start with 0
    for word, i in tqdm(abstract_word_index.items(), ncols=70):
        word = word.strip('\'')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_abstract[i] = embedding_vector
    # 得到embedding的单词占比
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix_abstract, axis=1))
    cover_per = nonzero_elements / (len(abstract_word_index) + 1)
    print("Abstract cover: {:.4f}".format(cover_per))

    claims_token = text.Tokenizer()
    claims_token.fit_on_texts(claims)
    claims_word_index = claims_token.word_index

    x_claims_train_seq = sequence.pad_sequences(claims_token.texts_to_sequences(x_claims_train), maxlen=1400)
    x_claims_valid_seq = sequence.pad_sequences(claims_token.texts_to_sequences(x_claims_valid), maxlen=1400)
    x_claims_test_seq = sequence.pad_sequences(claims_token.texts_to_sequences(x_claims_test), maxlen=1400)

    embedding_matrix_claims = np.zeros((len(claims_word_index) + 1, 300)) # 50是词向量的维度,+1 is because the matrix indices start with 0
    for word, i in tqdm(claims_word_index.items(), ncols=70):
        word = word.strip('\'')
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_claims[i] = embedding_vector
    # 得到embedding的单词占比
    nonzero_elements = np.count_nonzero(np.count_nonzero(embedding_matrix_claims, axis=1))
    cover_per = nonzero_elements / (len(claims_word_index) + 1)
    print("Claims cover: {:.4f}".format(cover_per))
    
    # 5.模型
    EMBEDDING_DIM = embedding_matrix_abstract.shape[1]
    SEQ_LEN_abstract = x_abs_train_seq.shape[1]
    SEQ_LEN_claims = x_claims_train_seq.shape[1]
    VOCAB_SIZE_abstract = abstract_word_index
    VOCAB_SIZE_claims = claims_word_index

    # abstract_model, claims_model, fusion_model = fusion_network()
    fusion_model = fusion_network()
    print(fusion_model.summary())

    # fusion
    fusion_model_name = "1FusionModel-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"
    filepath = './' + fusion_model_name
    fusion_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')    
    fusion_history = fusion_model.fit(x = [x_abs_train_seq, x_claims_train_seq], y = y_category_train, 
                                    validation_data=([x_abs_valid_seq, x_claims_valid_seq], y_category_valid),
                                    verbose=1, epochs=Epoch, batch_size=BATCH_SIZE, shuffle=False, 
                                    callbacks=[fusion_checkpoint],class_weight=cw)
    # fusion_model.save("./fusion_model.hdf5")
    plot_history("fusion1", fusion_history)

    y_pred = fusion_model.predict([x_abs_test_seq,x_claims_test_seq], batch_size=BATCH_SIZE, verbose=0)
    for i in range(len(y_pred)):
        max_value=max(y_pred[i])
        for j in range(len(y_pred[i])):
            if max_value==y_pred[i][j]:
                y_pred[i][j]=1
            else:
                y_pred[i][j]=0
    report = classification_report(y_category_test, y_pred, digits=4)
    p = re.compile('  |\n', re.S)
    report = p.sub(' ', report)
    metrics_content = re.findall("([\d]{1}\.[\d]{4})    777", report)
    with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/fusion_result.txt", 'a') as f:
        f.write("Result:\n")
        f.write("acc=" + metrics_content[0])
        f.write("\tmacro=" + metrics_content[1])
        f.write("\tweighted=" + metrics_content[2])
        f.write("\tEpochs=" + str(Epoch))
        f.write("\n---------------------\n")

    # # abstract
    # abstract_model_name = "AbsModel-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"
    # filepath = './' + abstract_model_name
    # abs_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # abs_history = abstract_model.fit(x_abs_train_seq, y_category_train, 
    #                                 validation_data=(x_abs_valid_seq, y_category_valid), 
    #                                 verbose=1, epochs=Epoch, batch_size=BATCH_SIZE, shuffle=False, 
    #                                 callbacks=[abs_checkpoint], class_weight=cw)
    # # abstract_model.save("./abstract_model.hdf5")
    # plot_history("abstract", abs_history)

    # y_pred = abstract_model.predict(x_abs_test_seq, batch_size=BATCH_SIZE, verbose=0)
    # for i in range(len(y_pred)):
    #     max_value=max(y_pred[i])
    #     for j in range(len(y_pred[i])):
    #         if max_value==y_pred[i][j]:
    #             y_pred[i][j]=1
    #         else:
    #             y_pred[i][j]=0
    # report = classification_report(y_category_test, y_pred, digits=4)
    # p = re.compile('  |\n', re.S)
    # report = p.sub(' ', report)
    # metrics_content = re.findall("([\d]{1}\.[\d]{4})    777", report)
    # with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/abstract_result.txt", 'a') as f:
    #     f.write("Result:\n")
    #     f.write("acc=" + metrics_content[0])
    #     f.write("\tmacro=" + metrics_content[1])
    #     f.write("\tweighted=" + metrics_content[2])
    #     f.write("\tEpochs=" + str(Epoch))
    #     f.write("\n---------------------\n")
    
    # # claims
    # claims_model_name = "ClaimsModel-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5"
    # filepath = './' + claims_model_name
    # claims_checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    # claims_history = claims_model.fit(x_claims_train_seq, y_category_train, 
    #                                 validation_data=(x_claims_valid_seq, y_category_valid),
    #                                 verbose=1, epochs=Epoch, batch_size=BATCH_SIZE, shuffle=False, 
    #                                 callbacks=[claims_checkpoint], class_weight=cw)
    # # claims_model.save("./claims_model.hdf5")
    # plot_history("claims1", claims_history)

    # y_pred = claims_model.predict(x_claims_test_seq, batch_size=BATCH_SIZE, verbose=0)
    # for i in range(len(y_pred)):
    #     max_value=max(y_pred[i])
    #     for j in range(len(y_pred[i])):
    #         if max_value==y_pred[i][j]:
    #             y_pred[i][j]=1
    #         else:
    #             y_pred[i][j]=0
    # report = classification_report(y_category_test, y_pred, digits=4)
    # p = re.compile('  |\n', re.S)
    # report = p.sub(' ', report)
    # metrics_content = re.findall("([\d]{1}\.[\d]{4})    777", report)
    # with open("/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/claims_result.txt", 'a') as f:
    #     f.write("Result:\n")
    #     f.write("acc=" + metrics_content[0])
    #     f.write("\tmacro=" + metrics_content[1])
    #     f.write("\tweighted=" + metrics_content[2])
    #     f.write("\tEpochs=" + str(Epoch))
    #     f.write("\n---------------------\n")