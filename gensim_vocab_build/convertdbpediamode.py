from gensim import *
# model = models.KeyedVectors.load_word2vec_format('en.model', binary=True)
model = models.Word2Vec.load('dbpedia.model')

model.wv.save_word2vec_format('dbpedia.txt')
print ("done")