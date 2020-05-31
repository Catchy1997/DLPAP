import gensim
from gensim.test.utils import  datapath,get_tmpfile
from gensim.scripts.glove2word2vec import glove2word2vec
 
def transfer(glovedilepath, word2vecfilepath):
    gloveFile=datapath(glovedilepath)
    word2vecFile=get_tmpfile(word2vecfilepath)
    glove2word2vec(gloveFile,word2vecFile)

glovedilepath = "D:/Pythonworkspace/Word Embedding/GloVe/vector/glove_42B/glove.42B.300d.txt"
word2vecfilepath = "D:/Pythonworkspace/Word Embedding/GloVe/vector/glove_42B/glove.42B.300d.word2vec.txt"
transfer(glovedilepath, word2vecfilepath)