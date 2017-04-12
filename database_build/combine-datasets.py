import re, sys, requests, time, random, bottleneck, json, os, spacy, ast
from multiprocessing import Pool
from functools import partial
from itertools import islice
import numpy as np

from query_vector_converter import *
from graph_edit_distance import *
from approximate_selectivity import get_selectivity

def multireplace(string, replacements):
    """
    Given a string and a replacement map, it returns the replaced string.
    :param str string: string to execute replacements on
    :param dict replacements: replacement dictionary {value to find: value to replace}
    :rtype: str
    """
    # Place longer ones first to keep shorter substrings from matching where the longer ones should take place
    # For instance given the replacements {'ab': 'AB', 'abc': 'ABC'} against the string 'hey abc', it should produce
    # 'hey ABC' and not 'hey ABc'
    substrs = sorted(replacements, key=len, reverse=True)

    # Create a big OR regex that matches any of the substrings to replace
    regexp = re.compile('|'.join(map(re.escape, substrs)))

    # For each match, look up the new string in the replacements
    return regexp.sub(lambda match: replacements[match.group(0)], string)

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

def extract_word2vec(line, nlp):
    splitted = line.split('\t')
    query = re.sub('[(){}<>|=]', '', splitted[0])  

    query = re.sub(r'(SELECT|FROM|PREFIX|OPTIONAL|FILTER|WHERE|ORDER)', r' \1', query)
    query = re.sub(r'http://', '', query)
    query = re.sub(r'/', ' ', query)
    query_meaning = extract_meaning(query)
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
    # return similar[0][:24] !!
    return query_vec.vector.tolist()


def gen_query_vectors(log_file):
    results = []

    nlp = spacy.load('en', vectors='vec')
    count = 0
    with open(log_file) as in_:
        vectors_done = []
        for l_ in in_:
            meaning_vec = extract_word2vec(l_, nlp)
            similarity_vec = similarity_list(meaning_vec, vectors_done)
            # vectors_done.append(meaning_vec) !!

            query_line = l_.strip('\n')
            query_line = query_line.split('\t')
            query_vec = unicode(query_line[1])
            query_vec = ast.literal_eval(query_vec)

            res_vec = query_vec + meaning_vec
            res = (query_line[0] + '\t' + str(res_vec) + '\t' + str(query_line[2]) + '\t' + str(query_line[3]) + '\n')
            if res is not None:
                count +=1
                print ".%d." % (count)
                sys.stdout.flush()
                results.append(res)

    with open(log_file + '-nosim', 'a') as out:
        for entry in results:
        # for entry in results.get():
            if entry is not None:
                out.write(str(entry))

def main():
    print "hi"
    reload(sys)
    sys.setdefaultencoding('utf-8')
    log_file = 'random200k.log-rnn'
    gen_query_vectors(log_file)

if __name__ == '__main__':
    main()