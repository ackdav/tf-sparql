# import modules & set up logging
import gensim, logging, re, os

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class SPARQLsentences(object):
    def __init__(self, dirname):
        self.dirname = dirname
 
    def __iter__(self):
        for fname in os.listdir(self.dirname):
            for line in open(os.path.join(self.dirname, fname)):
                line = re.sub('[(){}<>|=.]', '', line)
                line = " ".join(line.split())
                yield line.split()

# model.save('/tmp/sparqlmodel')
# model = gensim.models.Word2Vec.load('/tmp/sparqlmodel')

sentences = SPARQLsentences('./sentences') # a memory-friendly iterator
model = gensim.models.Word2Vec(sentences, min_count=1, size=200, workers=4) #min_count => how many times a word has to come up

model.save('./tmp/sparqlmodel.model')
model.wv.save_word2vec_format('./tmp/sparqlmodel.txt')