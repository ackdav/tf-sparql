import re, sys, requests, time, random, bottleneck, json, os, spacy, ast
from multiprocessing import Pool
import numpy as np
from tqdm import *

from database_helper import *

def extract_meaning(query):
    query = re.sub(r'(:)', r'\1 ', query)

    replacements = {'SELECT': '', 'FROM': '','PREFIX': '','OPTIONAL': '','FILTER': '','WHERE': '','ORDER': '',
                        'DESCRIBE': '','DISTINCT': '','IN': '','FILTER': '','LIMIT': '', 'dbpedia.org': '', 'de.dbpedia.org': '',
                        '_': ' ', '&&': '','UNION': '', 'VALUES': '', 'Select':'', 'select': '', 'describe': '', '[': '', ']': '',
                        'OFFSET': '', 'distinct': '', 'define': '', 'CONSTRUCT': '', 'resource': '', 'AS': '', 'GROUP BY': '', 'abstract':'', 'label': '', 'ontology': '', 'Filter': '', 'where': '', '-mode': '', 'BY': '', 'en': '', 'fr': '',
                        'contains': '', 'subject': '', 'property': '', 'sameAs': '', 'foaf': '', 'langMatches': '', 'lat': '', 'long': '',
                        'type': ''}
    query = multireplace(query, replacements)
    query = ''.join([i for i in query if not i.isdigit()])
    query = re.sub('[!*.,";\'&]', '', query)
    query = re.sub(r'(@)', r' \1', query)
    query = query.split()
    cleaned_query = []
    for en in query:
        if ':' not in en and 'www' not in en and '?' not in en and '@' not in en and '#' not in en and '\\' not in en:
            cleaned_query.append(en)
    query = " ".join(cleaned_query)
    return query

def extract_word2vec(query_str, nlp):
    query = re.sub('[(){}<>|=]', '', query_str)  

    query = re.sub(r'(SELECT|FROM|PREFIX|OPTIONAL|FILTER|WHERE|ORDER)', r' \1', query)
    query = re.sub(r'http://', '', query)
    query = re.sub(r'/', ' ', query)
    # query_meaning = extract_meaning(query)
    query_meaning = query
    query_meaning_vec = nlp(unicode(query_meaning))
    return query_meaning_vec

def similarity_list(query_vec, vectors_done):
    similarities = []
    for q_v in vectors_done:
        sim = float(query_vec.similarity(q_v))
        similarities.append(sim)
    #Pad vector to 24-size for the first 23 queries
    if len(similarities) < 24:
        similarities_pad = similarities + [0.] * (24 - len(similarities)) # pad to length
    else:
        similarities_pad = similarities
    similarities_pad = np.asarray([similarities_pad])
    biggest_n_values = -bottleneck.partition(-similarities_pad, 23)[:23]
    similar = biggest_n_values.tolist()
    return similar[0][:24]

def gen_query_vectors(log_file):
    results = []

    nlp = spacy.load('en', vectors='vec')
    count = 0
    with open(log_file) as in_, tqdm(total=linecount(log_file)) as pbar:
        vectors_done = []
        for l_ in in_:
            query_line = l_.strip('\n')
            query_line = query_line.split('\t')
            query_str = query_line[0]
            time_vec = query_line[1]
            query_structure = query_line[2]
            query_ged = query_line[3]
            time_warm = query_line[4]
            time_cold = query_line[5]
            res_size = query_line[6]

            meaning_vec = extract_word2vec(query_str, nlp)
            similarity_vec = similarity_list(meaning_vec, vectors_done)
            vectors_done.append(meaning_vec)

            query_structure = unicode(query_structure)
            query_structure = ast.literal_eval(query_structure)
            query_ged = unicode(query_ged)
            query_ged = ast.literal_eval(query_ged)

            res = (str(time_vec) + '\t' + str(query_structure) + '\t' + str(query_ged) + '\t' + str(similarity_vec) + '\t' + str(meaning_vec.vector.tolist())+ '\t' + str(time_warm) + '\t' + str(time_cold) + '\t' + str(res_size) + '\n')
            if res is not None:
                pbar.update(1)
                sys.stdout.flush()
                results.append(res)

    with open(log_file + '-complete', 'a') as out:
        for entry in results:
        # for entry in results.get():
            if entry is not None:
                out.write(str(entry))

def main():
    '''These methods compute the word2vec and 
    a similarity vector between the current vec and the past vectors.
    '''
    reload(sys)
    sys.setdefaultencoding('utf-8')
    log_file = 'database-time.log'
    gen_query_vectors(log_file)

if __name__ == '__main__':
    main()