import re, sys, requests, time, random
from multiprocessing.dummy import Pool 
from functools import partial
from itertools import islice

from sparql_graph.query_vector_converter import *
from graph_edit_distance import get_distances

ged_samples = []
counter = 0

def clean_query_helper(query):
	strip = query.strip('\n')
	strip = strip.replace('"', '\\"')
	split = strip.split('\t')
	query = split[0]
	#insert spaces before sparql cmds
	query = re.sub(r'([A-Z]{3,})', r' \1', query) 
	query = ' '.join(query.split())
	time = split[1]
	result_size = split[2]	
	return (query, time, result_size)

def prepare_ged_benchmark_queries(log_file, num_bench):
	global ged_samples

	# with open('dbpedia-benchmark-queries.txt') as f:
	# 	for query in f:
	# 		ged_samples.append(clean_query_helper(query)[0])
	# print ged_samples
	# get random sample of ged-distance benchmark queries
	with open(log_file) as f:
		lines = random.sample(f.readlines(), num_bench)
	for query in lines:
		ged_samples.append(clean_query_helper(query)[0])
	print ged_samples
	return ged_samples

def convert_query_graph(line):
	global counter
	query, time, result_size = clean_query_helper(line)
	structure_vector = structural_query_vector(query)
	ged_distances = get_distances(query, ged_samples)

	if structure_vector != -1 and len(ged_distances)==len(ged_samples):
		counter += 1
		print counter
		# insert time at end, if converting the db and not ged-sample-set

		query_vec = structure_vector + ged_distances
		query_vec.insert(len(query_vec), time)
		# write db
		# print '.',
		sys.stdout.flush()
		return (query + '\t' + str(query_vec) + "\t" + str(result_size) + '\n')

def gen_query_vectors(log_file):
	# open queries and regex for links
	results = []
	with open(log_file) as f:
		pool = Pool(4)
		results = pool.map(convert_query_graph, f, 1)
		# for line in f:
		# 	results.append(preprocess_write_db(True, line))
	with open(log_file + '-out', 'a') as out:
		for entry in results:
			out.write(str(entry))

def main():
	print "hi"
	log_file = 'db-cold-novec-1k.txt'
	ged_samples = prepare_ged_benchmark_queries(log_file, 8) # EXPAND
	gen_query_vectors(log_file)

if __name__ == '__main__':
	main()