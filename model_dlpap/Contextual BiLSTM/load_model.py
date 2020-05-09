import pandas as pd
import numpy as np
from tqdm import tqdm
import os,re

import xgboost
from sklearn import preprocessing,linear_model, metrics, ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,average_precision_score
from sklearn.metrics import precision_recall_curve,roc_curve,auc
from sklearn.metrics import classification_report
from sklearn.utils import class_weight

import tensorflow as tf
from keras import backend as K
from keras.preprocessing import text, sequence
from keras.layers.merge import concatenate
from keras import layers, models
from keras.utils import to_categorical
from keras.utils import np_utils

filename = "/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/after_process.xlsx"
data = pd.read_excel(filename,encoding='utf-8')

abstract = data['abstract_final']
claims = data['claims_final']

abstract_token = text.Tokenizer()
abstract_token.fit_on_texts(abstract)
abstract_word_index = abstract_token.word_index

claims_token = text.Tokenizer()
claims_token.fit_on_texts(claims)
claims_word_index = claims_token.word_index

x_abs_seq = sequence.pad_sequences(abstract_token.texts_to_sequences(abstract), maxlen=140)
x_claims_seq = sequence.pad_sequences(claims_token.texts_to_sequences(claims), maxlen=1400)

filepath = "/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/model/"
docLabels = [f for f in os.listdir(filepath) if f.endswith('.h5')]

for model_filename in tqdm(docLabels, ncols=70):
    if len(re.findall(r'AbsModel', model_filename)) > 0:
        model = models.load_model(model_filename)
        # model.summary()
        # get_3rd_layer_output = K.function(inputs=[model.layers[0].input,model.layers[1].input],outputs=[model.layers[-2].output])
        # layer_output = get_3rd_layer_output([x_abs_train_seq, x_claims_train_seq])[0]
        # layer_output.shape
        layer_model = models.Model(inputs=[model.layers[0].input],outputs=[model.layers[-3].output])
        intermediate_output = layer_model.predict(x = x_abs_seq, batch_size=256, verbose=1)
    if len(re.findall(r'ClaimsModel', model_filename)) > 0:
        model = models.load_model(model_filename)
        layer_model = models.Model(inputs=[model.layers[0].input],outputs=[model.layers[-3].output])
        intermediate_output = layer_model.predict(x = x_claims_seq, batch_size=256, verbose=1)
    if len(re.findall(r'FusionModel', model_filename)) > 0:
        model = models.load_model(model_filename)
        layer_model = models.Model(inputs=[model.layers[0].input,model.layers[1].input],outputs=[model.layers[-3].output])
        intermediate_output = layer_model.predict(x = [x_abs_seq, x_claims_seq], batch_size=256, verbose=1)
    df = pd.DataFrame(intermediate_output)
    df.to_excel(model_filename+".xlsx")