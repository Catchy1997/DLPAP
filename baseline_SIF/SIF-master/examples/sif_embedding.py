import pandas as pd
from tqdm import tqdm
import nltk.tokenize as tk
import sys
sys.path.append('../src')
import data_io, params, SIF_embedding

if __name__ == '__main__':
    # input
    wordfile = '../data/glove.6B.300d.txt' # word vector file
    weightfile = '../auxiliary_data/enwiki_vocab_min200.txt' # each line is a word and its frequency
    weightpara = 1e-3 # the parameter in the SIF weighting scheme, usually in the range [3e-5, 3e-3]
    rmpc = 1 # number of principal components to remove in SIF weighting scheme

    # set parameters
    params = params.params()
    params.rmpc = rmpc

    # load word vectors
    (words, We) = data_io.getWordmap(wordfile)

    # load word weights
    word2weight = data_io.getWordWeight(weightfile, weightpara) # word2weight['str'] is the weight for the word 'str'
    weight4ind = data_io.getWeight(words, word2weight) # weight4ind[i] is the weight for the i-th word

    # load sentences
    data = pd.read_excel("E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/patent_text_2010.xlsx", encoding='utf-8')
    index = 0
    for abstract in tqdm(data['claims'], ncols=70): # abstract&claims
        filename = "E:/Pythonworkspace/patent/process_data/sample3_G-06-F-17/textual/SIF/claims/" + str(index) + ".xlsx"
        tokens = tk.sent_tokenize(abstract)
        token_list = []
        for token in tokens:
            if len(token) < 3:
                pass
            else:
                token_list.append(token)
        # 将token_list输入
        x, m = data_io.sentences2idx(token_list, words) # x is the array of word indices, m is the binary mask indicating whether there is a word in that location
        w = data_io.seq2weight(x, m, weight4ind) # get word weights

        # get SIF embedding
        embedding = SIF_embedding.SIF_embedding(We, x, w, params) # embedding[i,:] is the embedding for sentence i
        df = pd.DataFrame(embedding)
        df.to_excel(filename, sheet_name='Sheet1')
        index = index + 1