import re, sys, requests, time, random
from multiprocessing import Pool
from functools import partial
from itertools import islice

from query_vector_converter import *
from graph_edit_distance import *
from approximate_selectivity import get_selectivity

ged_samples = []

def clean_query_helper(query):
	strip = query.strip('\n')
	strip = strip.replace('"', '\\"')
	split = strip.split('\t')
	query = split[0]
	#insert spaces before sparql cmds
	query = re.sub(r'(SELECT|FROM|PREFIX|OPTIONAL|FILTER)', r' \1', query)
	#escaped " dont work with jena
	query = re.sub(r'\\([!:])', r'\1', query)
	#all queries are missing virtuoso prefixes
	query = 'PREFIX dbpedia-owl: <http://dbpedia.org/ontology/> ' + query
	query = ' '.join(query.split())
	time_warm = split[1]
	time_cold = split[2]
	result_size = split[3]
	return (query, time_warm, time_cold, result_size)

def convert_query_graph(line):
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
		selectivity = get_selectivity(query)
	except:
		print ('query-convertion err'), sys.exc_info()[0], line
		structure_vector = -1

	if structure_vector != -1:
		query = remove_temp_prefixes(query)
		# insert time at end, if converting the db and not ged-sample-set
		query_vec = structure_vector + ged_distances
		query_vec.insert(len(query_vec), selectivity)
		# query_vec.insert(len(query_vec), time)
		
		# write db
		# print '.',
		sys.stdout.flush()
		return (query + '\t' + str(query_vec) + "\t" + str(time_warm) + '\t' + str(time_cold) + '\t'+ str(result_size) + '\n')

def gen_query_vectors(log_file):
	# open queries and regex for links
	results = []
	t0 = time.clock()

	with open(log_file) as f:
		pool = Pool()
		results = pool.map_async(convert_query_graph, f, 1)
		pool.close()
		while not results.ready():
			remaining = results._number_left
			print "Waiting for", remaining, "tasks to complete..."
			sys.stdout.flush()
			time.sleep(15.0)
	print time.clock()-t0

		# for line in f:
		# 	results.append(preprocess_write_db(True, line))
	with open(log_file + '-out', 'a') as out:
		for entry in results.get():
			if entry is not None:
				out.write(str(entry))

def main():
	print "hi"
	log_file = 'random200k.log-out'
	gen_query_vectors(log_file)

if __name__ == '__main__':
	main()
