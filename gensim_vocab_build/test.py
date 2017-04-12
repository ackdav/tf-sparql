import spacy, re, os, sys, bottleneck, json
import numpy as np

def parse_json_results(res_json):
    '''
    probably one of the most inefficient methods ever written in python history
    I apologize. 
    '''
    jay = json.loads(res_json)
    res_parsed = []
    for key in jay:
        for k, v in key.items():
            vals = repr(v['value'])
            vals = re.sub(r'(:)', r'\1 ', vals)
            vals = re.sub(r'http://', '', vals)
            vals = re.sub(r'/', ' ', vals)

            replacements_res = {'dbpedia.org': '', 'de.dbpedia.org': '', 'resource': '', 'ontology': '', '_': ' ', 'xmlns.com': '', 'foaf': '', 'http:': '', 'wikiPageWikiLink': '', 'http': '', 'www.w3.org': '',  }
            vals_clean = multireplace(vals, replacements_res)
            vals_clean = re.sub('[!*.,";\'&]', '', vals_clean)
            for word in vals_clean.split():
                if (not word.isdigit()) and (':' not in word):
                    res_parsed.append(vals_clean)
            if len(res_parsed)>100:
                break
        else: # this mechanism breaks out of 2 for loops
            continue
        break
    res_parsed = " ".join(res_parsed)
    return res_parsed

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

def main():
    reload(sys)
    sys.setdefaultencoding('utf-8')

    nlp = spacy.load('en', vectors='vec')
    results = []
    with open('random200k.log-out') as in_:
        vectors_done = []
        count = 0
        for line in in_:
            if len(results) > 10000:
                break
            
            # line = res.sub('([0-9]*\.[0-9]+|[0-9]+)$', '', line)
            splitted = line.split('\t')
                                                                   #1. parsed query 
            query = re.sub('[(){}<>|=]', '', splitted[0])          #2. cold exec time
            exec_time = splitted[1]                                #3. result size
            res_unparsed = splitted[3]                             #4. result in json                 

            query = re.sub(r'(SELECT|FROM|PREFIX|OPTIONAL|FILTER|WHERE|ORDER)', r' \1', query)
            query = re.sub(r'http://', '', query)
            query = re.sub(r'/', ' ', query)
            res_parsed = parse_json_results(res_unparsed)
            
            query_meaning = extract_meaning(query)
            query_vec = nlp(unicode(query))
            query_meaning_vec = nlp(unicode(query_meaning))
            res_parsed = nlp(unicode(res_parsed))

            #Extract the 24 most similar queries in past queries to 
            #hopefully extract likelines that query is pre-cached
            similarities = []
            for q_v in vectors_done:
                sim = float(query_meaning_vec.similarity(q_v))
                similarities.append(sim)
            #Pad vector to 24-size for the first 23 queries
            if len(similarities) < 24:
                similarities_pad = similarities + [0.] * (24 - len(similarities)) # pad to length
            else:
                similarities_pad = similarities
            similarities_pad = np.asarray([similarities_pad])
            biggest_n_values = -bottleneck.partition(-similarities_pad, 23)[:23]
            similar = biggest_n_values.tolist()
            query_vec_final = query_vec.vector.tolist() + similar[0][:24]

            # print similar[0][:24]
            vectors_done.append(res_parsed) # gather vectors_done for similarities
            
            count += 1
            print str(count) + " ",
            results.append(str(query_vec_final) + '\t' + str(exec_time) + '\n')

    with open('dbpedia_rnn.log-all2mean', 'w+') as out_:
        for res in results:
            out_.write(res)

if __name__ == '__main__':
    main()

