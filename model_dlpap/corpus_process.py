import pandas as pd
import numpy as np
from tqdm import tqdm
import re
import matplotlib.pyplot as plt

from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet as wn
from collections import defaultdict
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn import svm
from sklearn.metrics import accuracy_score

def status(x) : 
    return pd.Series([x.count(),x.sum(),x.min(),x.idxmin(),x.quantile(.25),x.median(),
                      x.quantile(.75),x.quantile(.90),x.quantile(.95),x.quantile(.99),x.mean(),x.max(),x.idxmax(),x.mode(),x.mad(),x.var(),
                      x.std(),x.skew(),x.kurt()],index=['非空数','求和','最小值','最小值位置','25%分位数',
                    '中位数','75%分位数','90%分位数','95%分位数','99%分位数','均值','最大值','最大值位数','众数','平均绝对偏差','方差','标准差','偏度','峰度'])

def distribution():
	fig = plt.figure(figsize=(10,6))

	plt.subplot(221)
	plt.plot(df['len_abstract'].index, df['len_abstract'].values)
	plt.ylabel("number of words")
	plt.xlabel("abstract")

	plt.subplot(222)
	plt.plot(df['len_claims'].index, df['len_claims'].values)
	plt.ylabel("number of words")
	plt.xlabel("claims")

	plt.subplot(223)
	plt.plot(df['abstract_max_len'].index, df['abstract_max_len'].values)
	plt.ylabel("number of sentences")
	plt.xlabel("abstract")

	plt.subplot(224)
	plt.plot(df['claims_max_len'].index, df['claims_max_len'].values)
	plt.ylabel("number of sentences")
	plt.xlabel("claims")

	plt.show()

def result_distribution():
	fig = plt.figure(figsize=(10,6))

	plt.subplot(221)
	df1 = df[df['result']==1]
	plt.scatter(df1.index, df1['len_abstract'], label="success")
	df2 = df[df['result']==0]
	plt.scatter(df2.index, df2['len_abstract'], label="fail")
	plt.ylabel("number of words")
	plt.xlabel("abstract")

	plt.subplot(222)
	df1 = df[df['result']==1]
	plt.scatter(df1.index, df1['len_claims'], label="success")
	df2 = df[df['result']==0]
	plt.scatter(df2.index, df2['len_claims'], label="fail")
	plt.ylabel("number of words")
	plt.xlabel("claims")

	plt.subplot(223)
	df1 = df[df['result']==1]
	plt.scatter(df1.index, df1['abstract_max_len'], label="success")
	df2 = df[df['result']==0]
	plt.scatter(df2.index, df2['abstract_max_len'], label="fail")
	plt.ylabel("number of sentences")
	plt.xlabel("abstract")

	plt.subplot(224)
	df1 = df[df['result']==1]
	plt.scatter(df1.index, df1['claims_max_len'], label="success")
	df2 = df[df['result']==0]
	plt.scatter(df2.index, df2['claims_max_len'], label="fail")
	plt.ylabel("number of sentences")
	plt.xlabel("claims")

	plt.legend()
	plt.grid()
	plt.show()


if __name__ == '__main__':
	data = pd.read_excel(r"E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/patent_text_2010.xlsx", encoding='utf-8')
	print("number of patents: "+str(len(data)))

	# sent_tokenize
	abstract_max_len = []
	for index,abstract in enumerate(tqdm(data['abstract'], ncols=70)):
    	tokens = sent_tokenize(abstract)
    	token_list = []
    	for token in tokens:
        	if len(token) < 3:
            	pass
        	else:
            	token_list.append(token)
    	abstract_max_len.append(len(token_list))
    	data.loc[index,'abstract_sen_count'] = len(token_list)

    claims_max_len = []
    for index,claims in enumerate(tqdm(data['claims'], ncols=70)):
        tokens = sent_tokenize(claims)
        token_list = []
        for token in tokens:
            if len(token) < 3:
                pass
            else:
                token_list.append(token)
        claims_max_len.append(len(token_list))
        data.loc[index,'claims_sen_count'] = len(token_list)

    # word_tokenize
    # -abstract
    data['abstract'].dropna(inplace=True)
    # data['abstract'] = [entry.lower() for entry in data['abstract']]
    data['abstract'] = [word_tokenize(entry) for entry in data['abstract']]

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    len_abstract = []
    for index,entry in enumerate(tqdm(data['abstract'], ncols=70)):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        or word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]]) # 对这三类做词形还原
                Final_words.append(word_Final)
        len_abstract.append(len(Final_words))
        data.loc[index,'abstract_final'] = str(Final_words)
        data.loc[index,'abstract_count'] = len(Final_words)
    
    # -claims
    data['claims'].dropna(inplace=True)
    # data['claims'] = [entry.lower() for entry in data['claims']]
    data['claims'] = [word_tokenize(entry) for entry in data['claims']]

    tag_map = defaultdict(lambda : wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    len_claims = []
    for index,entry in enumerate(tqdm(data['claims'], ncols=70)):
        Final_words = []
        word_Lemmatized = WordNetLemmatizer()
        for word, tag in pos_tag(entry):
            if word not in stopwords.words('english') and word.isalpha():
                word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]]) # 对这三类做词形还原
                Final_words.append(word_Final)
        len_claims.append(len(Final_words))
        data.loc[index,'claims_final'] = str(Final_words)
        data.loc[index,'claims_count'] = len(Final_words)

    # save the corpus
    data.to_excel("E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/corpus_process/after_process.xlsx", encoding='utf-8')
    
    # statistic
    df = pd.DataFrame(np.array([len_abstract,len_claims,abstract_max_len,claims_max_len,data['result']]).T, columns=['len_abstract','len_claims','abstract_max_len','claims_max_len','result'])
    df.apply(status)

    # matplotlib
    distribution()
    result_distribution()