import re, sys, requests, time, random, bottleneck, json, os, spacy
from multiprocessing import Pool
from functools import partial
from itertools import islice
import numpy as np

from query_vector_converter import *
from graph_edit_distance import *
# from approximate_selectivity import get_selectivity

ged_samples = []

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

def clean_query_helper(query):
	strip = query.strip('\n')
	strip = strip.replace('"', '\\"')
	split = strip.split('\t')
	query = split[0]
	#insert spaces before sparql cmds
	query = re.sub(r'(SELECT|FROM|PREFIX|OPTIONAL|FILTER|WHERE|ORDER)', r' \1', query)
	#escaped " dont work with jena
	query = re.sub(r'\\([!:])', r'\1', query)
	#all queries are missing virtuoso prefixes
	query = 'PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> ' + query
	query = ' '.join(query.split())
	# time_warm = split[1]
	time_warm = None
	time_cold = split[1] # split[2]
	
	result_size = split[2] # split[3]
	return (query, time_warm, time_cold, result_size)

def convert_query_graph(line, similarities):
	try:
		query, time_warm, time_cold, result_size = clean_query_helper(line)
		if 'DESCRIBE' in query:
			query = rewrite_describe_queries(query)
		query = add_missing_prefixes(query)

	except:
		print ('query-preparation err'), sys.exc_info()[0], line
		query = ''

	try:
		structure_vector = structural_query_vector(query)
		ged_distances = get_distances(query)

		# selectivity = get_selectivity(query)
	except:
		print ('query-convertion err'), sys.exc_info()[0], line
		structure_vector = -1

	if structure_vector != -1:
		query = remove_temp_prefixes(query)
		# insert time at end, if converting the db and not ged-sample-set
		query_vec = structure_vector + ged_distances + similarities
		# query_vec.insert(len(query_vec), similarities)
		

		# write db
		# return (query + '\t' + str(query_vec) + "\t" + str(time_warm) + '\t' + str(time_cold) + '\t'+ str(result_size) + '\n')
		return (query + '\t' + str(query_vec) + '\t' + str(time_cold) + '\t'+ str(result_size) + '\n')

def extract_word2vec(line, nlp):
	splitted = line.split('\t')
	query = re.sub('[(){}<>|=]', '', splitted[0])  
	exec_time = splitted[1]                        
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
	return similar[0][:24]

def gen_query_vectors(log_file):
	'''
	Main method of converting a query-string into the query vector, used in the tensorflow model
	vector consists of:
	1. structural query features
	2. graph edit distances to benchmark queries
	3. selectivity approximation of query
	'''
	# open queries and regex for links
	results = []
	# with open(log_file) as f:
	# 	pool = Pool()
	# 	results = pool.map_async(convert_query_graph, f, 1)
	# 	pool.close()
	# 	while not results.ready():
	# 		remaining = results._number_left
	# 		print "Waiting for", remaining, "tasks to complete..."
	# 		sys.stdout.flush()
	# 		time.sleep(15.0)
	nlp = spacy.load('en', vectors='vec')

	with open(log_file) as in_:
		vectors_done = []
		for l_ in in_:
			meaning_vec = extract_word2vec(l_, nlp)
			similarity_vec = similarity_list(meaning_vec, vectors_done)
			vectors_done.append(meaning_vec)

			res = convert_query_graph(l_, similarity_vec)
			if res is not None:
				print ".",
				sys.stdout.flush()
				results.append(res)

	with open(log_file + '-out', 'a') as out:
		for entry in results:
		# for entry in results.get():
			if entry is not None:
				out.write(str(entry))

def main():
	print "hi"
	reload(sys)
	sys.setdefaultencoding('utf-8')
	log_file = 'random200k.log-incl'
	gen_query_vectors(log_file)

if __name__ == '__main__':
	main()
