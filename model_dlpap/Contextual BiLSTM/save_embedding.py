import pickle
import numpy as np
from tqdm import tqdm

def save_obj(obj):
    with open('./embedding.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
	embeddings_index = {}
	for line in tqdm(open('/home/hxjiang/Pythonworkspace/patent/sample3_G-06-F-17/textual/glove.42B.300d.word2vec.txt', 'r', encoding='UTF-8')):
		values= line.split()
		embeddings_index[values[0]] = np.asarray(values[1:], dtype='float32')
	
	save_obj(embeddings_index)
	print("save embedding!")