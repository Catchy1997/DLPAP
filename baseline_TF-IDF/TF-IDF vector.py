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

if __name__ == '__main__':
	data = pd.read_excel(r"E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/corpus_process/after_process.xlsx",encoding='utf-8')

	# abstract
	Tfidf_vect_abs = TfidfVectorizer(max_features=300)
	Tfidf_vect_abs.fit(data['abstract_final'])
	Tfidf_feature_abs = Tfidf_vect_abs.transform(data['abstract_final'])
	df_abs = pd.DataFrame(Tfidf_feature_abs.toarray())
	df_abs.to_excel("E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/TF-IDF/abstract_TF-IDF_300d.xlsx")

	# claims
	Tfidf_vect_claims = TfidfVectorizer(max_features=300)
	Tfidf_vect_claims.fit(data['claims_final'])
	Tfidf_feature_claims = Tfidf_vect_claims.transform(data['claims_final'])
	df_claims = pd.DataFrame(Tfidf_feature_claims.toarray())
	df_claims.to_excel("E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/TF-IDF/claims_TF-IDF_300d.xlsx")